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
from datetime import datetime, timezone, timedelta

import operator
import re
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

from models import ClassifyResult, BatchCitationSupport, DraftAnswer, Intent, Score
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

# Secondary fallback for `_rewrite_pdf_query` when the LLM rewrite call
# itself fails — strips common conversational wrappers so vector search
# against a paper's full text isn't polluted with e.g. "Hi can you tell me
# what the paper says about..." Ordered so greetings/lead-ins are stripped
# before trailing "?" punctuation.
_PDF_QUERY_FILLER_PATTERNS = [
    re.compile(r"^\s*(hi|hello|hey)[,!.\s]+", re.IGNORECASE),
    re.compile(r"^\s*(so|well|ok(ay)?)[,!.\s]+", re.IGNORECASE),
    re.compile(r"\b(can|could|would)\s+you\s+(please\s+)?(tell|let)\s+me\s+", re.IGNORECASE),
    re.compile(r"\b(can|could|would)\s+you\s+(please\s+)?(explain|describe|summarize|find|search(\s+for)?|look\s*up)\s+", re.IGNORECASE),
    re.compile(r"\bi\s+(want|would like|need)\s+to\s+know\s+(about\s+)?", re.IGNORECASE),
    re.compile(r"\bwhat\s+does\s+(the|this)\s+paper\s+say\s+about\s+", re.IGNORECASE),
    re.compile(r"\btell\s+me\s+about\s+", re.IGNORECASE),
    re.compile(r"\bplease\s+", re.IGNORECASE),
    re.compile(r"[?!]+\s*$"),
]


def _strip_conversational_filler(text: str) -> str:
    """Best-effort regex cleanup of conversational wrapping around a
    question. Falls back to the original text unchanged if stripping would
    leave nothing behind, so the caller never ends up searching on ''."""
    cleaned = text
    for pattern in _PDF_QUERY_FILLER_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = cleaned.strip(" ,.!?")
    return cleaned if cleaned else text


BYPASS_INTENTS = {
    Intent.AUTHOR_LOOKUP.value,
    Intent.RECENT_DIGEST.value,
    Intent.CITATION_GRAPH.value,
    Intent.PAPER_LOOKUP.value,
    Intent.PAPER_QA.value,
}

warnings.filterwarnings("ignore")

IST = timezone(timedelta(hours=5, minutes=30))

def ist_converter(sec=None):
    if sec is None:
        sec = datetime.now().timestamp()
    # Convert the raw timestamp to IST, then to a time-tuple that logging expects
    return datetime.fromtimestamp(sec, tz=IST).timetuple()

logging.getLogger().setLevel(logging.CRITICAL)
logging.Formatter.converter = ist_converter

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
    per_paper_answers: List[Dict[str, Any]] # Map-reduce PAPER_QA (multi-paper comparisons)
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

    # Tools whose failure/emptiness is worth retrying against a different
    # provider before giving up on the retrieval step entirely.
    _FALLBACK_CHAIN = {"search_semantic_scholar": "search_papers"}

    async def retrieve(self, tool_name: str, tool_args: Dict[str, Any]) -> List[Document]:
        tool_fn = self._tool_registry.get(tool_name)
        if not tool_fn:
            return []

        raw_results, failed = await self._call_tool(tool_fn, tool_name, tool_args)
        effective_tool = tool_name

        # Chain-of-search: a Semantic Scholar rate limit (429) or an empty
        # result set shouldn't dead-end the retrieval — arXiv's own search
        # API covers most of the same academic-paper ground for a plain
        # keyword query, so retry there before surfacing "no results".
        fallback_name = self._FALLBACK_CHAIN.get(tool_name)
        if fallback_name and (failed or not raw_results):
            fallback_fn = self._tool_registry.get(fallback_name)
            if fallback_fn:
                fallback_args: Dict[str, Any] = {"query": tool_args.get("query", "")}
                if "max_results" in tool_args:
                    fallback_args["max_results"] = tool_args["max_results"]
                fb_raw, fb_failed = await self._call_tool(fallback_fn, fallback_name, fallback_args)
                if not fb_failed and fb_raw:
                    raw_results, effective_tool = fb_raw, fallback_name

        return self._normalize(raw_results, effective_tool)

    @staticmethod
    async def _call_tool(tool_fn: Any, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[Any, bool]:
        """Runs a sync tool in a background thread. Never raises — returns
        (result, failed) so callers can branch on a failure vs. a genuinely
        empty-but-successful result without a second try/except."""
        try:
            # Dispatch synchronous tools into background threads
            return await asyncio.to_thread(tool_fn, **tool_args), False
        except Exception as e:
            error_logger.error(f"tool call {tool_name}({tool_args}) failed: {e}", exc_info=True)
            return None, True

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
    def __init__(
        self,
        llm,
        threshold: float = 0.3,
        filterable_intents: Optional[set] = None,
        sleep_between_calls: float = 1.0,
        max_docs: Optional[int] = None,
    ):
        self._threshold = threshold
        self._filterable_intents = filterable_intents or {Intent.OPEN_ENDED.value}
        self._sleep_between_calls = sleep_between_calls
        self._max_docs = max_docs

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a STRICT retrieval evaluator for a scientific paper-search RAG pipeline.
Your task is to score how RELEVANT the provided paper chunk is to the user's question on a scale of 0.0 to 1.0.

SCORING RUBRIC:
- 0.8 to 1.0: HIGHLY relevant. The chunk directly answers the core question or provides critical requested details.
- 0.4 to 0.7: PARTIALLY relevant. The chunk discusses related topics but does not directly answer the main question.
- 0.0 to 0.3: IRRELEVANT. The chunk is off-topic or merely shares generic keywords without meaningful context.

RULES:
1. Be highly conservative with scores above 0.7.
2. Evaluate based purely on the semantic overlap of concepts, not just keyword matching.
3. Do NOT explain your reasoning. Output ONLY the evaluation score."""),
            ("human", "Question: {question}\n\nPaper:\n{chunk}"),
        ])
        self._score_chain = prompt | llm.with_structured_output(Score)

    async def refine(self, question: str, intent: str, docs: List[Document]) -> Tuple[List[Document], Dict[str, Dict[str, Any]], str]:
        good_docs = await self._filter_relevant(question, docs) if intent in self._filterable_intents else list(docs)
        # Scored docs arrive already sorted best-first.
        if self._max_docs is not None and len(good_docs) > self._max_docs:
            good_docs = good_docs[: self._max_docs]
        sources, refined_context = self._attribute(good_docs)
        return good_docs, sources, refined_context

    async def _filter_relevant(self, question: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return []
        scored: List[Tuple[Document, float]] = []
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
                scored.append((doc, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in scored]

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
                "content": doc.page_content,
            }
            lines.append(f"[{label}] {doc.page_content}")
        return sources, "\n\n".join(lines)


class AnswerGenerator:
    def __init__(self, llm):
        self._llm = llm

    async def generate(self, question: str, context: str, strict: bool = False) -> str:
        system = (
            "You are an expert academic research assistant.\n"
            "Your task is to synthesize a comprehensive, accurate answer to the user's question using ONLY the provided sources.\n\n"
            
            "CRITICAL INSTRUCTIONS:\n"
            "1. NO HALLUCINATION: If the answer cannot be found in the provided sources, explicitly state: 'The provided sources do not contain sufficient information to answer this question.' Do not guess.\n"
            "2. MANDATORY CITATIONS: Every factual claim, statistic, or methodological detail MUST be immediately followed by a citation marker corresponding to the source (e.g., [1], [2]).\n"
            "3. SYNTHESIS: Combine insights from multiple sources where appropriate rather than listing them out one by one.\n"
            "4. TONE: Maintain an objective, scholarly tone."
        )
        if strict:
            system += "\n5. STRICT CITATION MATCHING: Only use marker numbers that explicitly appear in the provided sources. Do not invent citation numbers."

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
    _MARKER_RE = re.compile(r"\[(\d+)\]")
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, llm, max_concurrency: int = 3):
        self._semaphore = asyncio.Semaphore(max_concurrency)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a STRICT fact-checker for a citation-verification pipeline.
You will be given a list of CLAIMS taken from a generated answer, along with the SOURCE text cited for each claim.

Determine whether EACH source factually supports its corresponding claim.

RULES:
1. Be strict: topical overlap or a merely related subject does NOT count as support.
2. The SOURCE does not need to state the claim word-for-word, but the specific fact/number/conclusion in the CLAIM must be traceable to the SOURCE.
3. If the CLAIM is a general statement not really asserting anything checkable (e.g. a transition sentence), treat it as supported.
4. Evaluate each claim independently."""),
            ("human", "Here are the claims to verify:\n\n{batched_claims}"),
        ])
        self._support_chain = prompt | llm.with_structured_output(BatchCitationSupport)

    async def verify(self, answer: str, sources: Dict[str, Dict[str, Any]], batch_size: int = 5) -> Tuple[bool, str]:
        used = set(self._MARKER_RE.findall(answer))
        valid = set(sources.keys())
        invalid = used - valid

        if invalid:
            return False, f"Unverified citation marker(s): {sorted(invalid)}"
        if not used and sources:
            return False, "Answer cites no sources despite sources being available."
        if not sources:
            return True, "No sources to verify against."

        unsupported = await self._check_semantic_support(answer, sources, batch_size)
        if unsupported:
            return False, f"Claim(s) not factually supported by their cited source(s): {sorted(unsupported)}"
        return True, "All citations verified against retrieved sources."

    async def _check_semantic_support(self, answer: str, sources: Dict[str, Dict[str, Any]], batch_size: int) -> List[str]:
        tasks = []
        claims_to_check = []
        for label, claim in self._extract_claims(answer):
            source_text = sources.get(label, {}).get("content") or sources.get(label, {}).get("title", "")

            if source_text:
                claims_to_check.append((label, claim, source_text))

        if not claims_to_check:
            return []

        for batch_idx  in range(0, len(claims_to_check), batch_size) :
            batched_text = ""
            for item_idx, (label, claim, source) in enumerate(claims_to_check[batch_idx  : batch_idx  + batch_size], 1):
                batched_text += f"--- ITEM {item_idx} ---\nLABEL: {label}\nCLAIM: {claim}\nSOURCE:\n{source}\n\n"
            task = asyncio.create_task(self._support_chain.ainvoke({"batched_claims": batched_text}))
            tasks.append(task)
    
        try:
            result = await asyncio.gather(*tasks, return_exceptions=True)

            unsupported: List[str] = []
            for res in result :
                if isinstance(res, Exception) :
                    continue
                for item in res.results :
                    if not item.supported and item.label not in unsupported:
                            unsupported.append(item.label)
                        
            return unsupported

        except Exception as e:
            error_logger.error(f"Error during batch semantic citation check: {e}", exc_info=True)
            # Fail open if the LLM crashes so we don't break the whole app
            return []

    @classmethod
    def _extract_claims(cls, answer: str) -> List[Tuple[str, str]]:
        """Pairs each citation marker with the sentence immediately before
        it — a best-effort proxy for 'the claim right before the marker'.
        Adjacent markers (e.g. "...finding [1][2].") share the same
        preceding sentence, since both are citing it."""
        claims: List[Tuple[str, str]] = []
        cursor = 0
        last_sentence = ""
        for match in cls._MARKER_RE.finditer(answer):
            segment = answer[cursor:match.start()]
            sentences = [s for s in cls._SENTENCE_SPLIT_RE.split(segment) if s.strip()]
            if sentences:
                last_sentence = sentences[-1].strip()
            if last_sentence:
                claims.append((match.group(1), last_sentence))
            cursor = match.end()
        return claims


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
            ("system", """You are a helpful, conversational research assistant.
Your task is to answer the user's question directly based on the ongoing conversation history.

CONSTRAINTS:
1. Rely entirely on the provided conversation history.
2. Keep your response concise and conversational.
3. NEVER claim to have searched ArXiv, Semantic Scholar, or any external database in this response.
4. If the user asks a new question that requires scientific literature, politely inform them that you will need to search the database for that."""),
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
    MAX_CONTEXT_DOCS = 8 # Chunks/papers kept per answer after relevance filtering

    # Intents whose retrieved docs are large/noisy enough to need semantic
    # filtering before they reach `generate`
    FILTERABLE_INTENTS = {Intent.OPEN_ENDED.value, Intent.PAPER_QA.value, Intent.RECENT_DIGEST.value}

    def __init__(self, session_id: str, api_keys: dict = None):
        self.session_id = session_id
        self.api_keys = api_keys or {}
        
        # Standalone checkpointer for the CLI
        self.checkpointer = MemorySaver()

        self.llm = build_llm(LLM_CONFIG_PATH)
        self._retriever = PaperRetriever(TOOL_REGISTRY)
        self._refiner = RelevanceRefiner(
            self.llm,
            threshold=self.LOWER_THRESHOLD,
            filterable_intents=self.FILTERABLE_INTENTS,
            max_docs=self.MAX_CONTEXT_DOCS,
        )
        self._generator = AnswerGenerator(self.llm)
        self._verifier = CitationVerifier(self.llm)
        self._context_responder = ContextResponder(self.llm, max_history_turns=self.HISTORY_WINDOW)
        self._pdf_rag = PdfRagService(
            embeddings=build_embeddings(EMBEDDING_PROVIDER, EMBEDDING_MODEL),
        )

        self.app = self._build_graph()

    def _push_status(self, step: str, message: str) -> None:
        """Terminal-friendly status display."""
        print(f" ⚙️  [{step}] {message}")

    def save_graph(self, filename: str = "docs/graph.png"):
        """Saves the LangGraph architecture as a PNG image."""
        try:
            # fetches the image bytes
            self.app.get_graph().draw_mermaid_png(output_file_path=filename)
                
            print(f"✅ Graph successfully saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save graph: {e}")

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
        workflow.add_node("paper_qa_map_reduce", self.paper_qa_map_reduce)
        workflow.add_node("synthesize_comparison", self.synthesize_comparison)
        workflow.add_node("generate", self.generate)
        workflow.add_node("verify", self.verify)
        workflow.add_node("finalize", self.finalize)

        workflow.add_edge(START, "classify")
        workflow.add_conditional_edges("classify", self._route_after_classify, {"clarify": "clarify", "context_reply": "context_reply", "build_query": "build_query", "draft_answer": "draft_answer", "no_results": "no_results", "paper_qa_map_reduce": "paper_qa_map_reduce"})
        workflow.add_edge("clarify", "classify")

        workflow.add_edge("draft_answer", "context_hint")
        workflow.add_conditional_edges("context_hint", self._route_after_context_hint, {"finalize_from_draft": "finalize_from_draft", "build_query": "build_query"})

        workflow.add_edge("build_query", "retrieve")
        workflow.add_edge("retrieve", "refine")
        workflow.add_conditional_edges("refine", self._route_after_refine, {"generate": "generate", "no_results": "no_results"})

        # Map-reduce path for multi-paper PAPER_QA comparisons
        workflow.add_conditional_edges("paper_qa_map_reduce", self._route_after_map_reduce, {"synthesize_comparison": "synthesize_comparison", "no_results": "no_results"})
        workflow.add_edge("synthesize_comparison", "verify")

        workflow.add_edge("generate", "verify")
        workflow.add_conditional_edges("verify", self._route_after_verify, {"generate": "generate", "finalize": "finalize"})
        
        workflow.add_edge("context_reply", END)
        workflow.add_edge("finalize_from_draft", END)
        workflow.add_edge("no_results", END)
        workflow.add_edge("finalize", END)

        return workflow.compile(checkpointer=self.checkpointer)

    def _route_after_classify(self, state: State) -> str:
        intent = state["intent"]
        
        # 1. Handle casual chat immediately
        if intent == Intent.GENERAL_CHAT.value: return "context_reply"

         # 2. If it's ambiguous, ask the user to clarify FIRST
        if state.get("ambiguous"):
            if state["clarification_rounds"] < self.MAX_CLARIFICATION_ROUNDS:
                return "clarify"
            else:
                return "no_results"

        # 3. Multi-paper PAPER_QA comparisons
        if self._find_multi_paper_qa_action(state):
            return "paper_qa_map_reduce"

        # 4. If it's clear and specific, go straight to tools
        if intent in BYPASS_INTENTS or len(state.get("actions", [])) > 1: return "build_query"

        # 5. If it's clear but OPEN_ENDED, check if the LLM knows it already (CRAG)
        return "draft_answer"

    def _route_after_context_hint(self, state: State) -> str:
        if state.get("context_hint") == "confident_draft" :
            return "finalize_from_draft"
        else :
            # The LLM doesn't know the answer confidently, so we MUST search the database
            return "build_query"

    def _route_after_refine(self, state: State) -> str:
        return "generate" if state.get("refined_context", "").strip() else "no_results"

    def _route_after_map_reduce(self, state: State) -> str:
        return "synthesize_comparison" if state.get("per_paper_answers") else "no_results"

    def _route_after_verify(self, state: State) -> str:
        if state.get("verified") or state.get("generation_attempts", 0) >= self.MAX_GENERATION_ATTEMPTS:
            return "finalize"
        return "generate"

    @staticmethod
    def _find_multi_paper_qa_action(state: State) -> Dict[str, Any]:
        """Returns the PAPER_QA action naming more than one explicit paper
        (not the ["all"] session-wide search) when it is the ONLY action for
        this question — i.e. a standalone multi-paper comparison, not one
        branch of a larger multi-part question. A compound question like
        "compare paper A and B, and also find recent AI papers" still goes
        through the standard build_query fan-out unchanged: folding
        map-reduce into that concurrent multi-tool-call path too is out of
        scope here, since it'd mean merging map-reduce's per-paper answers
        back into a single refine/generate pass alongside the other
        action's docs — a bigger restructuring than requested."""
        actions = state.get("actions") or []
        if len(actions) != 1:
            return {}
        action = actions[0]
        if action.get("intent") != Intent.PAPER_QA.value:
            return {}
        paper_ids = action.get("paper_ids") or []
        normalized = [str(pid).strip().lower() for pid in paper_ids]
        if len(paper_ids) > 1 and "all" not in normalized:
            return action
        return {}

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
                    "per_paper_answers": [],
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
            "You are an expert intent routing assistant for a scientific literature RAG system.\n"
            "Your task is to classify the user's PRIMARY intent into exactly ONE of the categories below, and extract relevant slots.\n\n"
            
            "CATEGORIES:\n"
            "- PAPER_LOOKUP: Use when the user asks for metadata (abstract, authors, publication date, PDF link) of a SPECIFIC paper. Extract `paper_id` if a strict alphanumeric arXiv ID is given (e.g., '1706.03762'); otherwise extract the paper's natural-language name into `paper_title`.\n"
            "- PAPER_QA: Use when the user asks about the internal content, methodology, results, or wants to COMPARE specific papers. (Requires full-text). Extract slots: `paper_ids` (list of arXiv IDs, or [\"all\"] for session papers), `paper_title` (if the user names the paper in natural language instead of giving an ID), and `query` (focused search string to find inside the text).\n"
            "- AUTHOR_LOOKUP: Use when the user specifically searches for publications by a named author.\n"
            "- RECENT_DIGEST: Use when the user asks for new, recent, or latest developments within a category/timeframe.\n"
            "- CITATION_GRAPH: Use when the user asks about references (what this paper cites) or forward citations (what cites this paper).\n"
            "- GENERAL_CHAT: Use for casual conversation, greetings, or follow-ups that rely entirely on chat history without needing a database search.\n"
            "- OPEN_ENDED: STRICT FALLBACK. Use ONLY for broad, thematic literature searches lacking specific papers, authors, or timeframes.\n\n"
            
            "MULTI-PART INSTRUCTIONS:\n"
            "If the query contains independent requests (e.g., 'Summarize paper X and also find recent AI papers'), assign the main focus to the primary intent. "
            "Assign secondary requests to `additional_actions` with their respective intents and slots.\n\n"
            
            "CRITICAL RULES:\n"
            "1. NEVER use OPEN_ENDED if the user names a specific paper, author, or asks for recent papers.\n"
            "2. ALWAYS use PAPER_QA if the user asks to compare specific papers.\n"
            "3. Extract all available slots accurately based strictly on the user's exact wording.\n"
            "4. If the user provides a natural language name or title of a paper, extract it to `paper_title`. ONLY populate `paper_id` if the user provides a strict alphanumeric ID (e.g., '1706.03762')."
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
                "paper_id": obj.paper_id, "paper_title": obj.paper_title,
                "citation_direction": obj.citation_direction,
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
            ("system", """You are an expert AI answering questions based on your internal knowledge.
Provide a detailed draft answer to the user's question. 
Additionally, assess your confidence in this answer on a scale of 0.0 to 1.0.

CONFIDENCE RUBRIC:
- 0.9 to 1.0: You know this as established, undeniable fact (e.g., standard math formulas, highly famous papers).
- 0.6 to 0.8: You are fairly certain but might miss recent developments or specific nuances.
- 0.0 to 0.5: You are guessing, recalling vaguely, or the topic requires highly specific, recent literature retrieval."""),
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
            if action.get("paper_title"):
                return {"tool_name": "search_title", "tool_args": {"query": action["paper_title"]}}
            if action.get("paper_id"):
                tool_name = "get_paper_pdf_url" if any(kw in state["question"].lower() for kw in ("pdf", "download")) else "get_paper_details"
                return {"tool_name": tool_name, "tool_args": {"paper_id": action["paper_id"]}}
            # Neither slot populated (e.g. classify() failed and fell back to a
            # bare ClassifyResult) — keep the old, safe-but-empty behavior
            # rather than guessing.
            return {"tool_name": "get_paper_details", "tool_args": {"paper_id": None}}
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

    async def paper_qa_map_reduce(self, state: State) -> Dict[str, Any]:
        """Map step for a multi-paper PAPER_QA comparison: retrieve, filter,
        and generate an independent answer for each requested paper in
        parallel, so no single `generate` call ever has to hold more than
        one paper's full text at once. Each paper's answer becomes one
        citable "source" for the reduce step (synthesize_comparison)."""
        action = self._find_multi_paper_qa_action(state)
        paper_ids = action.get("paper_ids") or []
        query = action.get("query") or state["question"]
        self._push_status("paper_qa_map", f"Researching {len(paper_ids)} papers independently…")

        map_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a meticulous academic research assistant extracting factual information from a single scientific paper.
Your objective is to answer the user's query using ONLY the provided Paper Context.

STRICT RULES:
1. NO HALLUCINATION: Base your entire response solely on the provided text. Never use external knowledge, even if you know the paper well.
2. HONESTY: If the context lacks sufficient information to answer the query, output EXACTLY: "Insufficient information in the provided context." Do not guess, infer, or hallucinate.
3. ISOLATED EXTRACTION: If the query asks to compare this paper with another, ignore the missing paper. Extract ONLY the methodologies, results, and facts relevant to THIS paper so they can be compared later. Do not state that the other paper is missing.
4. CONCISENESS: Be direct, objective, and highly factual. Avoid conversational filler (e.g., "The paper states that...").
5. NO CITATION MARKERS: Do not include bracketed reference numbers (e.g., [1]) or author-year citations in your output.

Focus purely on extracting the most accurate and relevant details for the user's query."""),
            ("human", "User Query: {query}\n\nPaper Context:\n{context}")
        ])
        map_chain = map_prompt | self.llm | StrOutputParser()

        async def _answer_for_paper(paper_id: str) -> Optional[Dict[str, Any]]:
            docs = await asyncio.to_thread(
                self._pdf_rag.retrieve,
                paper_ids=[paper_id],
                query=query,
                on_status=self._push_status,
            )
            good_docs, sources, refined_context = await self._refiner.refine(
                query, Intent.PAPER_QA.value, docs
            )
            if not refined_context.strip():
                return None

            summary = await map_chain.ainvoke({"query": query, "context": refined_context})

            first_meta = next(iter(sources.values()), {})
            return {
                "paper_id": paper_id,
                "title": first_meta.get("title") or paper_id,
                "url": first_meta.get("url", ""),
                "summary": summary,
            }

        results = await asyncio.gather(
            *(_answer_for_paper(pid) for pid in paper_ids), return_exceptions=True
        )

        per_paper: List[Dict[str, Any]] = []
        for paper_id, res in zip(paper_ids, results):
            if isinstance(res, Exception):
                error_logger.error(f"map-reduce PAPER_QA failed for {paper_id}: {res}", exc_info=res)
                continue
            if res is not None:
                per_paper.append(res)

        return {"per_paper_answers": per_paper}

    async def synthesize_comparison(self, state: State) -> Dict[str, Any]:
        """Reduce step: combines the independently generated per-paper
        answers into one comparison. Each paper's summary becomes a single
        citable source ([1], [2], ...), which slots into the same
        `sources`/`refined_context` shape the standard single-paper path
        produces — so the existing CitationVerifier/finalize/regeneration-
        on-failure machinery works unchanged from here on."""
        self._push_status("synthesize_comparison", "Combining per-paper findings…")
        per_paper = state.get("per_paper_answers", [])

        sources: Dict[str, Dict[str, Any]] = {}
        lines: List[str] = []
        for i, entry in enumerate(per_paper, start=1):
            label = str(i)
            sources[label] = {
                "title": entry.get("title", ""),
                "url": entry.get("url", ""),
                "authors": "",
                "paper_id": entry.get("paper_id", ""),
                "content": entry.get("summary", ""),
            }
            lines.append(f"[{label}] {entry.get('summary', '')}")
        refined_context = "\n\n".join(lines)

        answer = await self._generator.generate(state["question"], refined_context)
        return {
            "sources": sources,
            "refined_context": refined_context,
            "generated_answer": answer,
            "generation_attempts": 1,
        }

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
        verified, notes = await self._verifier.verify(state.get("generated_answer", ""), state.get("sources", {}))
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
        chain = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful research assistant. The user's query is ambiguous. "
                       "Ask EXACTLY ONE concise, clarifying question to help narrow down their exact intent, preferred authors, timeframes, or specific papers. "
                       "Do NOT attempt to answer the user's query."), 
            ("human", "Question: {question}")
        ]) | self.llm | StrOutputParser()
        try:
            return (await chain.ainvoke({"question": state["question"]})).strip()
        except Exception as e:
            error_logger.error(f"Error generating clarifying question: {e}", exc_info=True)
            return "Could you clarify what specifically you'd like to know?"

    async def _rewrite_query(self, state: State) -> str:
        chain = ChatPromptTemplate.from_messages([
            ("system", "You are an expert search query optimizer. "
                       "Convert the user's question into a keyword-dense search query (6-14 words) for an academic database.\n"
                       "RULES:\n"
                       "1. Remove conversational filler (e.g., 'Can you find papers on...').\n"
                       "2. Focus strictly on core entities, methodologies, and technical terms.\n"
                       "3. Do NOT phrase it as a question. Output ONLY the optimized keywords."), 
            ("human", "Question: {question}")
        ]) | self.llm | StrOutputParser()
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
            ("system", "You are an expert at full-text document retrieval. "
                       "Convert the user's question into a highly focused search string (6-14 words) optimized for vector search WITHIN a specific academic paper.\n"
                       "RULES:\n"
                       "1. Focus on specific variables, methodology steps, results, or unique claims.\n"
                       "2. Remove generic academic filler (e.g., 'What does the paper say about...').\n"
                       "3. Output ONLY the optimized search string."), 
            ("human", "Question: {question}")
        ]) | self.llm | StrOutputParser()
        try:
            return (await chain.ainvoke({"question": state["question"]})).strip()
        except Exception as e:
            error_logger.error(f"Error rewriting pdf query: {e}", exc_info=True)
            return _strip_conversational_filler(state["question"])


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
    service.save_graph()

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