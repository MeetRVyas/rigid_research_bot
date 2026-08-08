import argparse
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

from models import ClassifyResult, DraftAnswer, Intent, Score
from llm_factory import build_llm
from config import LLM_CONFIG

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
}

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
    slots: Dict[str, Any]
    clarification_rounds: int
    clarification_answer: str
    draft_answer: str
    draft_confidence: float
    context_hint: str
    tool_name: str
    tool_args: Dict[str, Any]
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

    def retrieve(self, tool_name: str, tool_args: Dict[str, Any]) -> List[Document]:
        tool_fn = self._tool_registry.get(tool_name)
        if not tool_fn:
            return []
        try:
            raw_results = tool_fn(**tool_args)
        except Exception as e:
            print(f"[ERROR] tool call {tool_name}({tool_args}) failed: {e}")
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

    def refine(self, question: str, intent: str, docs: List[Document]) -> Tuple[List[Document], Dict[str, Dict[str, Any]], str]:
        good_docs = self._filter_relevant(question, docs) if intent in self._filterable_intents else docs
        sources, refined_context = self._attribute(good_docs)
        return good_docs, sources, refined_context

    def _filter_relevant(self, question: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return []
        good_docs: List[Document] = []
        for doc in docs:
            try:
                res: Score = self._score_chain.invoke({"question": question, "chunk": doc.page_content})
                score = res.score
            except Exception:
                score = 0.0
            finally:
                time.sleep(self._sleep_between_calls)

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

    def generate(self, question: str, context: str, strict: bool = False) -> str:
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
        return self._extract_text(chain.invoke({"question": question, "context": context}))

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

    def respond(self, question: str, history: List[Dict[str, str]]) -> str:
        try:
            return self._chain.invoke({
                "question": question,
                "history": self.format_history(history, self._max_history_turns),
            }).strip()
        except Exception:
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
    MAX_CLARIFICATION_ROUNDS = 1
    MAX_GENERATION_ATTEMPTS = 2
    DEFAULT_CATEGORY = "cs.AI"
    DEFAULT_RECENT_DAYS = 7
    HISTORY_WINDOW = 6

    def __init__(self, session_id: str, model_name: str, provider: str = "ollama", api_keys: dict = None):
        self.session_id = session_id
        self.api_keys = api_keys or {}
        
        # Standalone checkpointer for the CLI
        self.checkpointer = MemorySaver()

        self.llm = build_llm(LLM_CONFIG)
        self._retriever = PaperRetriever(TOOL_REGISTRY)
        self._refiner = RelevanceRefiner(self.llm, threshold=self.LOWER_THRESHOLD)
        self._generator = AnswerGenerator(self.llm)
        self._verifier = CitationVerifier()
        self._context_responder = ContextResponder(self.llm, max_history_turns=self.HISTORY_WINDOW)

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
        workflow.add_conditional_edges("classify", self._route_after_classify, {"build_query": "build_query", "clarify": "clarify", "context_reply": "context_reply"})
        workflow.add_edge("clarify", "draft_answer")
        workflow.add_edge("draft_answer", "context_hint")
        workflow.add_conditional_edges("context_hint", self._route_after_context_hint, {"build_query": "build_query", "finalize_from_draft": "finalize_from_draft"})
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
        if intent in BYPASS_INTENTS: return "build_query"
        return "clarify"

    def _route_after_context_hint(self, state: State) -> str:
        return "finalize_from_draft" if state.get("context_hint") == "confident_draft" else "build_query"

    def _route_after_refine(self, state: State) -> str:
        return "generate" if state.get("refined_context", "").strip() else "no_results"

    def _route_after_verify(self, state: State) -> str:
        if state.get("verified") or state.get("generation_attempts", 0) >= self.MAX_GENERATION_ATTEMPTS:
            return "finalize"
        return "generate"

    def run(self, question: Optional[str] = None, resume_answer: Optional[str] = None, thread_id: Optional[str] = None) -> dict:
        config = {"configurable": {"thread_id": thread_id or self.session_id}}
        try:
            if resume_answer is not None:
                result = self.app.invoke(Command(resume=resume_answer), config=config)
            else:
                if not question:
                    raise ValueError("`question` is required unless resuming.")
                result = self.app.invoke({
                    "question": question,
                    "original_question": question,
                    "intent": "", "intent_confidence": 0.0, "ambiguous": False, "slots": {},
                    "clarification_rounds": 0, "clarification_answer": "", "draft_answer": "", "draft_confidence": 0.0,
                    "context_hint": "", "tool_name": "", "tool_args": {}, "search_query": "",
                    "raw_docs": [], "good_docs": [], "sources": {}, "refined_context": "",
                    "generated_answer": "", "generation_attempts": 0, "answer": "",
                    "verified": False, "verification_notes": "", "answered_from_context": False,
                }, config=config)

            if result.get("__interrupt__"):
                payload = result["__interrupt__"][0].value
                return {"status": "needs_clarification", "question": payload.get("question", "Could you clarify?")}

            return {"status": "complete", **result}
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Pipeline crashed: {e}")

    # --- Nodes ---
    def classify(self, state: State) -> Dict[str, Any]:
        self._push_status("classify", "Classifying your question…")
        history = ContextResponder.format_history(state.get("chat_history", []), self.HISTORY_WINDOW)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a router. Classify into: GENERAL_CHAT, AUTHOR_LOOKUP, RECENT_DIGEST, CITATION_GRAPH, PAPER_LOOKUP, or OPEN_ENDED. Extract slots. Do NOT return schema or explain."),
            ("human", "Conversation so far:\n{history}\n\nQuestion: {question}"),
        ])
        chain = prompt | self.llm.with_structured_output(ClassifyResult)
        try:
            res: ClassifyResult = chain.invoke({"question": state["question"], "history": history})
        except Exception:
            res = ClassifyResult(intent=Intent.OPEN_ENDED, confidence=0.0, ambiguous=True)
        return {
            "intent": res.intent.value if isinstance(res.intent, Intent) else str(res.intent),
            "intent_confidence": res.confidence, "ambiguous": res.ambiguous,
            "slots": {"author_name": res.author_name, "category": res.category, "days": res.days, "paper_id": res.paper_id, "citation_direction": res.citation_direction},
        }

    def clarify(self, state: State) -> Dict[str, Any]:
        rounds = state.get("clarification_rounds", 0)
        if not state.get("ambiguous") or rounds >= self.MAX_CLARIFICATION_ROUNDS:
            return {}
        self._push_status("clarify", "Question needs clarification…")
        clarifying_question = self._generate_clarifying_question(state)
        answer = interrupt({"question": clarifying_question})
        return {
            "clarification_answer": answer, "clarification_rounds": rounds + 1,
            "question": f"{state['question']}\n\nAdditional context from user: {answer}",
        }

    def draft_answer(self, state: State) -> Dict[str, Any]:
        self._push_status("draft_answer", "Drafting a preliminary answer…")
        history = ContextResponder.format_history(state.get("chat_history", []), self.HISTORY_WINDOW)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question from general knowledge alone. Rate your confidence [0.0, 1.0]."),
            ("human", "Conversation:\n{history}\n\nQuestion: {question}"),
        ])
        chain = prompt | self.llm.with_structured_output(DraftAnswer)
        try:
            res: DraftAnswer = chain.invoke({"question": state["question"], "history": history})
        except Exception:
            res = DraftAnswer(answer="", confidence=0.0)
        return {"draft_answer": res.answer, "draft_confidence": res.confidence}

    def context_hint(self, state: State) -> Dict[str, Any]:
        hint = "confident_draft" if state.get("draft_confidence", 0.0) >= self.DRAFT_CONFIDENCE_THRESHOLD else "web_primed"
        self._push_status("context_hint", f"Context assessment: {hint}")
        return {"context_hint": hint}

    def context_reply(self, state: State) -> Dict[str, Any]:
        self._push_status("context_reply", "Answering from conversation context…")
        answer = self._context_responder.respond(state["question"], state.get("chat_history", []))
        return {"answer": answer, "verified": True, "verification_notes": "No search performed.", "answered_from_context": True, **self._history_delta(state, answer)}

    def finalize_from_draft(self, state: State) -> Dict[str, Any]:
        self._push_status("finalize_from_draft", "Answering from what I already know…")
        answer = state.get("draft_answer") or "I don't have a confident answer. Searching needed."
        return {"answer": answer, "verified": True, "verification_notes": "Draft used.", "answered_from_context": True, **self._history_delta(state, answer)}

    def build_query(self, state: State) -> Dict[str, Any]:
        self._push_status("build_query", "Selecting tool…")
        intent, slots = state["intent"], state.get("slots", {}) or {}
        if intent == Intent.AUTHOR_LOOKUP.value: return {"tool_name": "get_author_papers", "tool_args": {"author": slots.get("author_name") or state["question"]}}
        if intent == Intent.RECENT_DIGEST.value: return {"tool_name": "get_recent_papers", "tool_args": {"category": slots.get("category") or self.DEFAULT_CATEGORY, "days": slots.get("days") or self.DEFAULT_RECENT_DAYS}}
        if intent == Intent.CITATION_GRAPH.value:
            tool_name = "get_paper_citations" if (slots.get("citation_direction") or "incoming") == "incoming" else "get_related_papers"
            return {"tool_name": tool_name, "tool_args": {"paper_id": slots.get("paper_id")}}
        if intent == Intent.PAPER_LOOKUP.value:
            tool_name = "get_paper_pdf_url" if any(kw in state["question"].lower() for kw in ("pdf", "download")) else "get_paper_details"
            return {"tool_name": tool_name, "tool_args": {"paper_id": slots.get("paper_id")}}
        query = self._rewrite_query(state)
        return {"tool_name": "search_semantic_scholar", "tool_args": {"query": query}, "search_query": query}

    def retrieve(self, state: State) -> Dict[str, Any]:
        self._push_status("retrieve", "Fetching papers…")
        return {"raw_docs": self._retriever.retrieve(state.get("tool_name", ""), state.get("tool_args", {}) or {})}

    def refine(self, state: State) -> Dict[str, Any]:
        self._push_status("refine", "Filtering and attributing sources…")
        good_docs, sources, refined_context = self._refiner.refine(state["question"], state["intent"], state.get("raw_docs", []))
        return {"good_docs": good_docs, "sources": sources, "refined_context": refined_context}

    def no_results(self, state: State) -> Dict[str, Any]:
        self._push_status("no_results", "No relevant papers found…")
        answer = "I couldn't find enough relevant papers to answer this question."
        return {"answer": answer, "verified": False, "verification_notes": "no context retrieved", "answered_from_context": False, **self._history_delta(state, answer)}

    def generate(self, state: State) -> Dict[str, Any]:
        self._push_status("generate", "Generating your answer…")
        attempts = state.get("generation_attempts", 0)
        text = self._generator.generate(state["question"], state.get("refined_context", ""), strict=(attempts > 0))
        return {"generated_answer": text, "generation_attempts": attempts + 1}

    def verify(self, state: State) -> Dict[str, Any]:
        self._push_status("verify", "Checking citations…")
        verified, notes = self._verifier.verify(state.get("generated_answer", ""), state.get("sources", {}))
        return {"verified": verified, "verification_notes": notes}

    def finalize(self, state: State) -> Dict[str, Any]:
        self._push_status("finalize", "Finalizing your answer…")
        answer = state.get("generated_answer", "")
        if sources := state.get("sources", {}):
            answer = f"{answer}\n\n{ReferenceFormatter.format(sources)}"
        return {"answer": answer, "answered_from_context": False, **self._history_delta(state, answer)}

    # --- Helpers ---
    def _history_delta(self, state: State, final_answer: str) -> Dict[str, Any]:
        return {"chat_history": [{"role": "user", "content": state.get("original_question") or state["question"]}, {"role": "assistant", "content": final_answer}]}

    def _generate_clarifying_question(self, state: State) -> str:
        chain = ChatPromptTemplate.from_messages([("system", "Ask ONE specific clarifying question. Do not answer."), ("human", "Question: {question}")]) | self.llm | StrOutputParser()
        try:
            return chain.invoke({"question": state["question"]}).strip()
        except Exception:
            return "Could you clarify what specifically you'd like to know?"

    def _rewrite_query(self, state: State) -> str:
        chain = ChatPromptTemplate.from_messages([("system", "Rewrite into a concise 6-14 word search query. Do NOT answer."), ("human", "Question: {question}")]) | self.llm | StrOutputParser()
        try:
            return chain.invoke({"question": state["question"]}).strip()
        except Exception:
            return state["question"]


# ======================================================================
# Main CLI Loop
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Standalone CLI for CRAG ArXiv/Semantic-Scholar Research Assistant")
    parser.add_argument("--model", type=str, default="llama3", help="Model name to use (e.g., llama3, gpt-4o)")
    parser.add_argument("--provider", type=str, default="ollama", help="LLM Provider (e.g., ollama, openai, anthropic)")
    args = parser.parse_args()

    session_id = str(uuid.uuid4())
    print("=" * 60)
    print(f"🚀 Starting CLI RAG Session (ID: {session_id})")
    print(f"🧠 Provider: {args.provider.upper()} | Model: {args.model}")
    print("💡 Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    service = CRAG_Service(
        session_id=session_id,
        model_name=args.model,
        provider=args.provider
    )

    while True:
        try:
            user_input = input("\n🧑 User: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break

            result = service.run(question=user_input)

            while result.get("status") == "needs_clarification":
                clarification = input(f"\n🤖 Assistant: {result['question']}\n\n🧑 Clarify: ").strip()
                if clarification.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    return
                result = service.run(resume_answer=clarification)

            if result.get("status") == "complete":
                print(f"\n🤖 Assistant:\n{result.get('answer')}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ [Error] {e}")

    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()