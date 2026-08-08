import os
import asyncio
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule

import warnings
import logging

import operator
import re
import time
import traceback
import uuid
from typing import Annotated, Any, Dict, List, Optional, Tuple, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from dotenv import load_dotenv
load_dotenv()

from models import ClassifyResult, DraftAnswer, Intent, Score
from llm_factory import build_llm, build_embeddings
from config import LLM_CONFIG_PATH, EMBEDDING_PROVIDER, EMBEDDING_MODEL
from rag_service import PdfRagService

from arxiv_service import (
    batch_get_papers,
    get_author_papers,
    get_paper_citations,
    get_paper_details,
    get_paper_pdf_url,
    get_recent_papers,
    get_related_papers,
    search_abstract,
    search_by_category,
    search_papers,
    search_semantic_scholar,
    search_title,
)

TOOL_REGISTRY = {
    "search_papers": search_papers,
    "get_paper_details": get_paper_details,
    "get_paper_pdf_url": get_paper_pdf_url,
    "get_recent_papers": get_recent_papers,
    "get_related_papers": get_related_papers,
    "get_paper_citations": get_paper_citations,
    "get_author_papers": get_author_papers,
    "search_by_category": search_by_category,
    "search_title": search_title,
    "search_abstract": search_abstract,
    "batch_get_papers": batch_get_papers,
    "search_semantic_scholar": search_semantic_scholar,
}

BYPASS_INTENTS = {
    Intent.AUTHOR_LOOKUP.value,
    Intent.RECENT_DIGEST.value,
    Intent.CITATION_GRAPH.value,
    Intent.PAPER_LOOKUP.value,
    Intent.PAPER_QA.value,
}

warnings.filterwarnings("ignore")

# 2. Hide internal library error logs (like the [ERROR] tool call... message)
# This forces libraries to only print CRITICAL issues, hiding standard warnings/errors
logging.getLogger().setLevel(logging.CRITICAL)

# Graceful Error Logging Setup
os.makedirs("logs", exist_ok=True)
error_logger = logging.getLogger("crag_error_logger")
error_logger.setLevel(logging.ERROR)
error_logger.propagate = False
if not error_logger.handlers:
    file_handler = logging.FileHandler(os.path.join("logs", "error.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    error_logger.addHandler(file_handler)

# ======================================================================
# Graph state
# ======================================================================
class State(TypedDict):
    question: str
    original_question: str
    clarification_rounds: int
    chat_history: Annotated[List[Dict[str, str]], operator.add]
    intent: str
    intent_confidence: float
    ambiguous: bool
    actions: List[Dict[str, Any]]
    clarification_rounds: int
    clarification_answer: str
    draft_answer: str
    draft_confidence: float
    context_hint: str
    tool_calls: List[Dict[str, Any]]
    search_query: str
    raw_docs: List[Document]
    good_docs: List[Document]
    sources: Dict[str, Dict[str, Any]]
    refined_context: str
    generated_answer: str
    generation_attempts: int
    verified: bool
    verification_notes: str
    answer: str
    answered_from_context: bool


# ======================================================================
# Single-responsibility collaborators
# ======================================================================

class PaperRetriever:
    def __init__(self, tool_registry: Dict[str, Any]):
        self._tool_registry = tool_registry

    async def retrieve(self, tool_name: str, tool_args: Dict[str, Any]) -> List[Document]:
        tool_fn = self._tool_registry.get(tool_name)
        if not tool_fn:
            return []
        try:
            # Dispatch synchronous tools into background threads
            raw_results = await asyncio.to_thread(tool_fn, **tool_args)
        except Exception as e:
            error_logger.error(f"tool call {tool_name}({tool_args}) failed: {e}", exc_info=True)
            return []
        return self._normalize(raw_results, tool_name)

    @staticmethod
    def _normalize(raw: Any, tool_name: str) -> List[Document]:
        if raw is None:
            return []
        items = [raw] if isinstance(raw, dict) else raw if isinstance(raw, list) else [raw]

        docs: List[Document] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            paper_id = item.get("id") or item.get("paper_id") or item.get("arxiv_id", "")
            title = item.get("title", "")
            authors = item.get("authors", [])
            authors_str = ", ".join(a if isinstance(a, str) else a.get("name", "") for a in authors) if isinstance(authors, list) else str(authors)
            summary = item.get("summary") or item.get("abstract") or ""
            url = item.get("url") or item.get("pdf_url") or item.get("entry_id", "")
            published = item.get("published", "")

            content = (
                f"TITLE: {title}\n"
                f"AUTHORS: {authors_str}\n"
                f"PUBLISHED: {published}\n"
                f"ABSTRACT: {summary}"
            ).strip()

            docs.append(Document(
                page_content=content,
                metadata={
                    "paper_id": paper_id,
                    "title": title,
                    "authors": authors_str,
                    "url": url,
                    "published": published,
                    "source_tool": tool_name,
                },
            ))
        return docs


class RelevanceRefiner:
    def __init__(self, llm, threshold: float = 0.3, filterable_intents: Optional[set] = None, sleep_between_calls: float = 1.0):
        self._threshold = threshold
        self._filterable_intents = filterable_intents or {Intent.OPEN_ENDED.value}
        self._sleep_between_calls = sleep_between_calls

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a strict retrieval evaluator for a paper-search RAG pipeline.\n"
                       "Score how relevant this paper is to the question, in [0.0, 1.0].\n"
                       "Be conservative with high scores.\nDo NOT return schema. Do NOT explain."),
            ("human", "Question: {question}\n\nPaper:\n{chunk}"),
        ])
        self._score_chain = prompt | llm.with_structured_output(Score)

    async def refine(self, question: str, intent: str, docs: List[Document]) -> Tuple[List[Document], Dict[str, Dict[str, Any]], str]:
        good_docs = await self._filter_relevant(question, docs) if intent in self._filterable_intents else docs
        sources, refined_context = self._attribute(good_docs)
        return good_docs, sources, refined_context

    async def _filter_relevant(self, question: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return []
        good_docs: List[Document] = []
        for doc in docs:
            try:
                res: Score = await self._score_chain.ainvoke({"question": question, "chunk": doc.page_content})
                score = res.score
            except Exception as e:
                error_logger.error(f"Error during relevance scoring: {e}", exc_info=True)
                score = 0.0
            finally:
                await asyncio.sleep(self._sleep_between_calls)

            if score > self._threshold:
                good_docs.append(doc)
        return good_docs

    @staticmethod
    def _attribute(docs: List[Document]) -> Tuple[Dict[str, Dict[str, Any]], str]:
        sources: Dict[str, Dict[str, Any]] = {}
        lines: List[str] = []
        for i, doc in enumerate(docs, start=1):
            label = str(i)
            sources[label] = {
                "title": doc.metadata.get("title", ""),
                "url": doc.metadata.get("url", ""),
                "authors": doc.metadata.get("authors", ""),
                "paper_id": doc.metadata.get("paper_id", ""),
            }
            lines.append(f"[{label}] {doc.page_content}")
        return sources, "\n\n".join(lines)


class AnswerGenerator:
    def __init__(self, llm):
        self._llm = llm

    async def generate(self, question: str, context: str, strict: bool = False) -> str:
        system = (
            "You are a research assistant. Answer ONLY using the provided sources.\n"
            "Every factual claim MUST be followed by a citation marker like [1], [2].\n"
            "If the sources are insufficient, say so plainly instead of guessing."
        )
        if strict:
            system += "\nIMPORTANT: Only use marker numbers that actually appear in the sources."

        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Question: {question}\n\nSources:\n{context}"),
        ])
        chain = prompt | self._llm
        return self._extract_text(await chain.ainvoke({"question": question, "context": context}))

    @staticmethod
    def _extract_text(response) -> str:
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, list):
                return "".join(block.get("text", "").strip() if isinstance(block, dict) else str(block).strip() for block in content)
            return str(content)
        return str(response)


class CitationVerifier:
    @staticmethod
    def verify(answer: str, sources: Dict[str, Dict[str, Any]]) -> Tuple[bool, str]:
        used = set(re.findall(r"\[(\d+)\]", answer))
        valid = set(sources.keys())
        invalid = used - valid

        if invalid:
            return False, f"Unverified citation marker(s): {sorted(invalid)}"
        if not used and sources:
            return False, "Answer cites no sources despite sources being available."
        return True, "All citations verified against retrieved sources."


class ReferenceFormatter:
    @staticmethod
    def format(sources: Dict[str, Dict[str, Any]]) -> str:
        lines = ["References:"]
        for label, meta in sources.items():
            title = meta.get("title") or "Untitled"
            url = meta.get("url", "")
            lines.append(f"[{label}] {title}" + (f" — {url}" if url else ""))
        return "\n".join(lines)


class ContextResponder:
    def __init__(self, llm, max_history_turns: int = 6):
        self._max_history_turns = max_history_turns
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a friendly research-assistant. Answer directly using conversation history. Do NOT claim to have searched ArXiv."),
            ("human", "Conversation so far:\n{history}\n\nUser: {question}"),
        ])
        self._chain = prompt | llm | StrOutputParser()

    async def respond(self, question: str, history: List[Dict[str, str]]) -> str:
        try:
            return (await self._chain.ainvoke({
                "question": question,
                "history": self.format_history(history, self._max_history_turns),
            })).strip()
        except Exception as e:
            error_logger.error(f"Error in ContextResponder: {e}", exc_info=True)
            return "Sorry, I had trouble putting that together — could you rephrase?"

    @staticmethod
    def format_history(history: List[Dict[str, str]], max_turns: int = 6) -> str:
        if not history:
            return "(no prior turns)"
        return "\n".join(f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in history[-max_turns:])


# ======================================================================
# Standalone CLI CRAG Service
# ======================================================================

class CRAG_Service:
    LOWER_THRESHOLD = 0.3
    DRAFT_CONFIDENCE_THRESHOLD = 0.6
    MAX_CLARIFICATION_ROUNDS = 2
    MAX_GENERATION_ATTEMPTS = 2
    DEFAULT_CATEGORY = "cs.AI"
    DEFAULT_RECENT_DAYS = 7
    HISTORY_WINDOW = 6

    def __init__(self, session_id: str, api_keys: dict = None):
        self.session_id = session_id
        self.api_keys = api_keys or {}
        
        # Standalone checkpointer for the CLI
        self.checkpointer = MemorySaver()

        self.llm = build_llm(LLM_CONFIG_PATH)
        self._retriever = PaperRetriever(TOOL_REGISTRY)
        self._refiner = RelevanceRefiner(self.llm, threshold=self.LOWER_THRESHOLD)
        self._generator = AnswerGenerator(self.llm)
        self._verifier = CitationVerifier()
        self._context_responder = ContextResponder(self.llm, max_history_turns=self.HISTORY_WINDOW)
        self._pdf_rag = PdfRagService(
            embeddings=build_embeddings(EMBEDDING_PROVIDER, EMBEDDING_MODEL),
        )

        self.app = self._build_graph()

    def _push_status(self, step: str, message: str) -> None:
        """Terminal-friendly status display."""
        print(f" ⚙️  [{step}] {message}")

    def _build_graph(self):
        workflow = StateGraph(State)

        workflow.add_node("classify", self.classify)
        workflow.add_node("clarify", self.clarify)
        workflow.add_node("draft_answer", self.draft_answer)
        workflow.add_node("context_hint", self.context_hint)
        workflow.add_node("context_reply", self.context_reply)
        workflow.add_node("finalize_from_draft", self.finalize_from_draft)
        workflow.add_node("build_query", self.build_query)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("refine", self.refine)
        workflow.add_node("no_results", self.no_results)
        workflow.add_node("generate", self.generate)
        workflow.add_node("verify", self.verify)
        workflow.add_node("finalize", self.finalize)

        workflow.add_edge(START, "classify")
        workflow.add_conditional_edges("classify", self._route_after_classify, {"build_query": "build_query", "clarify": "clarify", "context_reply": "context_reply", "no_results": "no_results"})
        workflow.add_edge("clarify", "draft_answer")
        workflow.add_edge("draft_answer", "context_hint")
        workflow.add_conditional_edges("context_hint", self._route_after_context_hint, {"classify": "classify", "finalize_from_draft": "finalize_from_draft"})
        workflow.add_edge("build_query", "retrieve")
        workflow.add_edge("retrieve", "refine")
        workflow.add_conditional_edges("refine", self._route_after_refine, {"generate": "generate", "no_results": "no_results"})
        workflow.add_edge("generate", "verify")
        workflow.add_conditional_edges("verify", self._route_after_verify, {"generate": "generate", "finalize": "finalize"})
        
        workflow.add_edge("context_reply", END)
        workflow.add_edge("finalize_from_draft", END)
        workflow.add_edge("no_results", END)
        workflow.add_edge("finalize", END)

        return workflow.compile(checkpointer=self.checkpointer)

    def _route_after_classify(self, state: State) -> str:
        intent = state["intent"]
        if intent == Intent.GENERAL_CHAT.value: return "context_reply"
        # A question that decomposed into more than one action is, by
        # construction, a compound-but-structured request rather than an
        # ambiguous one — skip the clarify/draft-answer detour for it too.
        if intent in BYPASS_INTENTS or len(state.get("actions", [])) > 1: return "build_query"
        if state["clarification_rounds"] < self.MAX_CLARIFICATION_ROUNDS :
            return "clarify"
        return "no_results"

    def _route_after_context_hint(self, state: State) -> str:
        if state.get("context_hint") == "confident_draft" :
            return "finalize_from_draft"
        else :
            return "classify"

    def _route_after_refine(self, state: State) -> str:
        return "generate" if state.get("refined_context", "").strip() else "no_results"

    def _route_after_verify(self, state: State) -> str:
        if state.get("verified") or state.get("generation_attempts", 0) >= self.MAX_GENERATION_ATTEMPTS:
            return "finalize"
        return "generate"

    async def run(self, question: Optional[str] = None, resume_answer: Optional[str] = None, thread_id: Optional[str] = None) -> dict:
        config = {"configurable": {"thread_id": thread_id or self.session_id}}
        try:
            if resume_answer is not None:
                result = await self.app.ainvoke(Command(resume=resume_answer), config=config)
            else:
                if not question:
                    raise ValueError("`question` is required unless resuming.")
                result = await self.app.ainvoke({
                    "question": question,
                    "original_question": question,
                    "intent": "", "intent_confidence": 0.0, "ambiguous": False, "actions": [],
                    "clarification_rounds": 0, "clarification_answer": "", "draft_answer": "", "draft_confidence": 0.0,
                    "context_hint": "", "tool_calls": [], "search_query": "",
                    "raw_docs": [], "good_docs": [], "sources": {}, "refined_context": "",
                    "generated_answer": "", "generation_attempts": 0, "answer": "",
                    "verified": False, "verification_notes": "", "answered_from_context": False,
                }, config=config)

            if result.get("__interrupt__"):
                payload = result["__interrupt__"][0].value
                return {"status": "needs_clarification", "question": payload.get("question", "Could you clarify?")}

            return {"status": "complete", **result}
        except Exception as e:
            error_logger.error(f"Pipeline crashed: {e}", exc_info=True)
            raise RuntimeError(f"Pipeline crashed: {e}")

    # --- Nodes ---
    async def classify(self, state: State) -> Dict[str, Any]:
        self._push_status("classify", "Classifying your question…")
        history = ContextResponder.format_history(state.get("chat_history", []), self.HISTORY_WINDOW)
        prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are an expert routing assistant for a scientific paper-research tool. "
            "Your job is to classify the user's PRIMARY intent into exactly one of the following categories.\n\n"
            
            "CATEGORIES:\n"
            "- PAPER_LOOKUP: The user asks for high-level metadata (abstract, authors, publication date, or PDF link) for specific papers.\n"
            "- PAPER_QA: The user asks about the internal content, methodology, results, or specific details of one or more papers. "
            "**This includes comparing specific papers.** Require this if answering needs the FULL TEXT. "
            "Extract slots: `paper_ids` (list of arXiv IDs, or [\"all\"] to search all session papers) and `query` (focused search string of what to find inside the text).\n"
            "- AUTHOR_LOOKUP: The user wants to find papers written by a specific named person.\n"
            "- RECENT_DIGEST: The user asks for new, recent, or latest developments in a specific category or time window.\n"
            "- CITATION_GRAPH: The user asks what a specific paper cites (references) or what cites it (forward citations).\n"
            "- GENERAL_CHAT: Casual conversation, greetings, or questions that do NOT require searching the paper database at all.\n"
            "- OPEN_ENDED: STRICT FALLBACK. Use ONLY for extremely broad, thematic literature queries that lack specific papers, authors, or timeframes. "
            "If the user mentions a specific paper, method, or author, DO NOT USE THIS.\n\n"
            
            "MULTI-PART QUESTIONS:\n"
            "If the query has multiple independent parts, assign the main part to the primary intent and slots. "
            "Put every OTHER independent part as its own entry in `additional_actions` (with its own intent and slots). "
            "Example: A full-text question about one paper (PAPER_QA) plus a separate request for recent papers (RECENT_DIGEST).\n\n"
            
            "CRITICAL RULES:\n"
            "1. Never classify as OPEN_ENDED if the user names a paper, an author, or asks for recent papers.\n"
            "2. If the user asks to compare papers, use PAPER_QA.\n"
            "3. Extract slots for the primary intent. Do NOT return schema. Do NOT explain your reasoning."
            ),
            ("human", "Conversation so far:\n{history}\n\nQuestion: {question}"),
        ])
        chain = prompt | self.llm.with_structured_output(ClassifyResult)
        try:
            res: ClassifyResult = await chain.ainvoke({"question": state["question"], "history": history})
        except Exception as e:
            error_logger.error(f"Error classifying intent: {e}", exc_info=True)
            res = ClassifyResult(intent=Intent.OPEN_ENDED, confidence=0.0, ambiguous=True)

        def _slots(obj) -> Dict[str, Any]:
            return {
                "author_name": obj.author_name, "category": obj.category, "days": obj.days,
                "paper_id": obj.paper_id, "citation_direction": obj.citation_direction,
                "paper_ids": obj.paper_ids, "query": obj.query,
            }

        primary_intent = res.intent.value if isinstance(res.intent, Intent) else str(res.intent)
        actions = [{"intent": primary_intent, **_slots(res)}]
        for extra in res.additional_actions:
            extra_intent = extra.intent.value if isinstance(extra.intent, Intent) else str(extra.intent)
            actions.append({"intent": extra_intent, **_slots(extra)})

        return {
            "intent": primary_intent,
            "intent_confidence": res.confidence, "ambiguous": res.ambiguous,
            "actions": actions,
        }

    async def clarify(self, state: State) -> Dict[str, Any]:
        rounds = state.get("clarification_rounds", 0)
        if not state.get("ambiguous") or rounds >= self.MAX_CLARIFICATION_ROUNDS:
            return {}
        self._push_status("clarify", "Question needs clarification…")
        clarifying_question = await self._generate_clarifying_question(state)
        answer = interrupt({"question": clarifying_question})
        return {
            "clarification_answer": answer, "clarification_rounds": rounds + 1,
            "question": f"{state['question']}\n\nAdditional context from user: {answer}",
        }

    async def draft_answer(self, state: State) -> Dict[str, Any]:
        self._push_status("draft_answer", "Drafting a preliminary answer…")
        history = ContextResponder.format_history(state.get("chat_history", []), self.HISTORY_WINDOW)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question from general knowledge alone. Rate your confidence [0.0, 1.0]."),
            ("human", "Conversation:\n{history}\n\nQuestion: {question}"),
        ])
        chain = prompt | self.llm.with_structured_output(DraftAnswer)
        try:
            res: DraftAnswer = await chain.ainvoke({"question": state["question"], "history": history})
        except Exception as e:
            error_logger.error(f"Error drafting answer: {e}", exc_info=True)
            res = DraftAnswer(answer="", confidence=0.0)
        return {"draft_answer": res.answer, "draft_confidence": res.confidence}

    async def context_hint(self, state: State) -> Dict[str, Any]:
        hint = "confident_draft" if state.get("draft_confidence", 0.0) >= self.DRAFT_CONFIDENCE_THRESHOLD else "web_primed"
        self._push_status("context_hint", f"Context assessment: {hint}")
        return {"context_hint": hint}

    async def context_reply(self, state: State) -> Dict[str, Any]:
        self._push_status("context_reply", "Answering from conversation context…")
        answer = await self._context_responder.respond(state["question"], state.get("chat_history", []))
        return {"answer": answer, "verified": True, "verification_notes": "No search performed.", "answered_from_context": True, **self._history_delta(state, answer)}

    async def finalize_from_draft(self, state: State) -> Dict[str, Any]:
        self._push_status("finalize_from_draft", "Answering from what I already know…")
        answer = state.get("draft_answer") or "I don't have a confident answer. Searching needed."
        return {"answer": answer, "verified": True, "verification_notes": "Draft used.", "answered_from_context": True, **self._history_delta(state, answer)}

    async def build_query(self, state: State) -> Dict[str, Any]:
        actions = state.get("actions") or [{"intent": state["intent"]}]
        self._push_status("build_query", f"Selecting tool{'s' if len(actions) > 1 else ''}…")
        
        tool_calls = []
        for action in actions:
            tool_calls.append(await self._action_to_tool_call(state, action))
            
        search_query = next(
            (c["tool_args"].get("query") for c in tool_calls if c.get("tool_args", {}).get("query")), ""
        )
        return {"tool_calls": tool_calls, "search_query": search_query}

    async def _action_to_tool_call(self, state: State, action: Dict[str, Any]) -> Dict[str, Any]:
        intent = action.get("intent")
        if intent == Intent.AUTHOR_LOOKUP.value:
            return {"tool_name": "get_author_papers", "tool_args": {"author": action.get("author_name") or state["question"]}}
        if intent == Intent.RECENT_DIGEST.value:
            return {"tool_name": "get_recent_papers", "tool_args": {"category": action.get("category") or self.DEFAULT_CATEGORY, "days": action.get("days") or self.DEFAULT_RECENT_DAYS}}
        if intent == Intent.CITATION_GRAPH.value:
            tool_name = "get_paper_citations" if (action.get("citation_direction") or "incoming") == "incoming" else "get_related_papers"
            return {"tool_name": tool_name, "tool_args": {"paper_id": action.get("paper_id")}}
        if intent == Intent.PAPER_LOOKUP.value:
            tool_name = "get_paper_pdf_url" if any(kw in state["question"].lower() for kw in ("pdf", "download")) else "get_paper_details"
            return {"tool_name": tool_name, "tool_args": {"paper_id": action.get("paper_id")}}
        if intent == Intent.PAPER_QA.value:
            paper_ids = action.get("paper_ids") or ([action["paper_id"]] if action.get("paper_id") else ["all"])
            query = action.get("query") or await self._rewrite_pdf_query(state)
            return {"tool_name": "pdf_rag_search", "tool_args": {"paper_ids": paper_ids, "query": query}}
        query = action.get("query") or await self._rewrite_query(state)
        return {"tool_name": "search_semantic_scholar", "tool_args": {"query": query}}

    async def retrieve(self, state: State) -> Dict[str, Any]:
        tool_calls = state.get("tool_calls") or []
        self._push_status("retrieve", f"Fetching from {len(tool_calls)} source(s)…" if len(tool_calls) != 1 else "Fetching papers…")

        if len(tool_calls) <= 1:
            docs = await self._dispatch_tool_call(tool_calls[0]) if tool_calls else []
        else:
            # Independent, I/O-bound calls (HTTP lookups and/or PDF download +
            # indexing) — run them concurrently so one slow PAPER_QA action
            # doesn't serialize behind the others. Each call is isolated so
            # one failure doesn't drop the rest of the results.
            docs = []
            results = await asyncio.gather(*(self._dispatch_tool_call(call) for call in tool_calls), return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    error_logger.error(f"retrieval action failed: {res}", exc_info=res)
                else:
                    docs.extend(res)
        return {"raw_docs": docs}

    async def _dispatch_tool_call(self, call: Dict[str, Any]) -> List[Document]:
        tool_name, tool_args = call.get("tool_name", ""), call.get("tool_args", {}) or {}
        if tool_name == "pdf_rag_search":
            return await asyncio.to_thread(
                self._pdf_rag.retrieve,
                paper_ids=tool_args.get("paper_ids") or ["all"],
                query=tool_args.get("query", ""),
                on_status=self._push_status,
            )
        return await self._retriever.retrieve(tool_name, tool_args)

    async def refine(self, state: State) -> Dict[str, Any]:
        self._push_status("refine", "Filtering and attributing sources…")
        good_docs, sources, refined_context = await self._refiner.refine(state["question"], state["intent"], state.get("raw_docs", []))
        return {"good_docs": good_docs, "sources": sources, "refined_context": refined_context}

    async def no_results(self, state: State) -> Dict[str, Any]:
        self._push_status("no_results", "No relevant papers found…")
        answer = "I couldn't find enough relevant papers to answer this question."
        return {"answer": answer, "verified": False, "verification_notes": "no context retrieved", "answered_from_context": False, **self._history_delta(state, answer)}

    async def generate(self, state: State) -> Dict[str, Any]:
        self._push_status("generate", "Generating your answer…")
        attempts = state.get("generation_attempts", 0)
        text = await self._generator.generate(state["question"], state.get("refined_context", ""), strict=(attempts > 0))
        return {"generated_answer": text, "generation_attempts": attempts + 1}

    async def verify(self, state: State) -> Dict[str, Any]:
        self._push_status("verify", "Checking citations…")
        verified, notes = self._verifier.verify(state.get("generated_answer", ""), state.get("sources", {}))
        return {"verified": verified, "verification_notes": notes}

    async def finalize(self, state: State) -> Dict[str, Any]:
        self._push_status("finalize", "Finalizing your answer…")
        answer = state.get("generated_answer", "")
        if sources := state.get("sources", {}):
            answer = f"{answer}\n\n{ReferenceFormatter.format(sources)}"
        return {"answer": answer, "answered_from_context": False, **self._history_delta(state, answer)}

    # --- Helpers ---
    def _history_delta(self, state: State, final_answer: str) -> Dict[str, Any]:
        return {"chat_history": [{"role": "user", "content": state.get("original_question") or state["question"]}, {"role": "assistant", "content": final_answer}]}

    async def _generate_clarifying_question(self, state: State) -> str:
        chain = ChatPromptTemplate.from_messages([("system", "Ask ONE specific clarifying question. Do not answer."), ("human", "Question: {question}")]) | self.llm | StrOutputParser()
        try:
            return (await chain.ainvoke({"question": state["question"]})).strip()
        except Exception as e:
            error_logger.error(f"Error generating clarifying question: {e}", exc_info=True)
            return "Could you clarify what specifically you'd like to know?"

    async def _rewrite_query(self, state: State) -> str:
        chain = ChatPromptTemplate.from_messages([("system", "Rewrite into a concise 6-14 word search query. Do NOT answer."), ("human", "Question: {question}")]) | self.llm | StrOutputParser()
        try:
            return (await chain.ainvoke({"question": state["question"]})).strip()
        except Exception as e:
            error_logger.error(f"Error rewriting query: {e}", exc_info=True)
            return state["question"]

    async def _rewrite_pdf_query(self, state: State) -> str:
        """Fallback for PAPER_QA when classify() didn't already fill in a
        focused `query` — rewrites the question into a search string aimed at
        the paper's full text rather than at ArXiv/Semantic Scholar."""
        chain = ChatPromptTemplate.from_messages([
            ("system", "Rewrite into a focused 6-14 word search query for searching WITHIN a paper's full text. Do NOT answer."),
            ("human", "Question: {question}"),
        ]) | self.llm | StrOutputParser()
        try:
            return (await chain.ainvoke({"question": state["question"]})).strip()
        except Exception as e:
            error_logger.error(f"Error rewriting pdf query: {e}", exc_info=True)
            return state["question"]


# ======================================================================
# Main CLI Loop
# ======================================================================
async def main():
    session_id = str(uuid.uuid4())
    console = Console()

    # Chatbot Welcome Screen
    welcome_text = (
        "Hello! I am your AI Assistant. I can search documents and answer questions.\n"
        "💡 Press [bold green]Enter[/bold green] to send.\n"
        "💡 Type [bold red]'exit'[/bold red] or press [bold red]Ctrl+D[/bold red] to stop."
    )
    console.print(Panel(welcome_text, title=f"🤖 RAG Agent (Session: {session_id[:8]})", border_style="cyan"))

    history_file = os.path.join(os.path.expanduser("~"), ".crag_cli_history")
    session = PromptSession(history=FileHistory(history_file))

    service = CRAG_Service(session_id=session_id)

    while True:
        try:
            console.print(Rule(style="dim default"))
            
            user_input = (await session.prompt_async(HTML("\n<b><ansigreen>🧑 You: </ansigreen></b>"))).strip()
            
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break

            with console.status("[bold cyan]Agent is thinking...", spinner="bouncingBar"):
                result = await service.run(question=user_input)

            while result.get("status") == "needs_clarification":
                console.print(f"\n[bold yellow]🤖 Assistant:[/bold yellow] {result['question']}")
                
                clarification_msg = HTML("<b><ansiyellow>💬 Clarify: </ansiyellow></b>")
                clarification = (await session.prompt_async(clarification_msg)).strip()
                
                if clarification.lower() in ["exit", "quit"]:
                    console.print("\n[bold cyan]👋 Goodbye![/bold cyan]")
                    return
                
                with console.status("[bold cyan]Re-evaluating context...", spinner="bouncingBar"):
                    result = await service.run(resume_answer=clarification)

            if result.get("status") == "complete":
                console.print("\n[bold cyan]🤖 Assistant:[/bold cyan]")
                md_answer = Markdown(result.get('answer'))
                console.print(md_answer)
                console.print()

        except KeyboardInterrupt:
            console.print("\n[dim yellow](Press Ctrl+D or type 'exit' to quit)[/dim yellow]")
            continue
        except EOFError:
            break
        except Exception as e:
            # 3. Graceful Error Handling (hidden from user as generic assistant replies)
            error_logger.error(f"Unexpected session error: {e}", exc_info=True)
            error_msg = str(e)
            
            console.print("\n[bold cyan]🤖 Assistant:[/bold cyan]")
            
            # Check if it's the specific Semantic Scholar Rate Limit error
            if "429" in error_msg and "Semantic Scholar" in error_msg:
                console.print("[bold orange3]⏳ Rate Limit Reached[/bold orange3]")
                console.print(
                    "[bold orange3]It looks like the research database is a bit busy right now. "
                    "Please wait a few seconds and try your question again.[/bold orange3]"
                )
            # Generic fallback for other tool failures
            elif "tool call" in error_msg.lower() or "failed" in error_msg.lower():
                console.print("\n[bold yellow]⚠️ Search Service Issue[/bold yellow]")
                console.print(
                    "[bold yellow]I had some trouble searching for that information. Could you try rephrasing your query?[/bold yellow]"
                )
            # Completely unexpected errors
            else:
                console.print(
                    "[bold red]I'm sorry, I encountered a minor hiccup while processing your request. "
                    "Let's try something else.[/bold red]"
                )

    console.print("\n[bold cyan]👋 Session ended. Goodbye![/bold cyan]")

if __name__ == "__main__":
    asyncio.run(main())