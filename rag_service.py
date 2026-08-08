"""
Full-text PDF RAG for the Veritas pipeline.

Dependencies this adds beyond what main.py already uses:
    pip install pypdf faiss-cpu langchain-text-splitters langchain-community --break-system-packages
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from arxiv_service import get_paper_details, get_paper_pdf_url

logger = logging.getLogger(__name__)

StatusCallback = Optional[Callable[[str, str], None]]

_MAX_PDF_BYTES = 50 * 1024 * 1024  # safety cap — a normal paper is a few MB
_DOWNLOAD_TIMEOUT = 30  # seconds


# ---------------------------------------------------------------------
# HTTP session — retried, and politely rate-limited to arxiv.org's PDF host
# (separate from arxiv_service's own limiter, which only covers the
# export.arxiv.org query API, not raw PDF downloads).
# ---------------------------------------------------------------------
def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "Veritas-PdfRag/1.0 (research tool)"})
    return session


_session = _build_session()


class _RateLimiter:
    """Thread-safe 'no more than one call every `delay` seconds' limiter."""

    def __init__(self, delay: float):
        self._delay = delay
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._delay:
                time.sleep(self._delay - elapsed)
            self._last_call = time.monotonic()


_pdf_rate_limiter = _RateLimiter(delay=2.0)


# ---------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------
class PdfDownloader:
    def __init__(self, cache_dir: str = "./pdf_cache"):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self, paper_id: str) -> Optional[Path]:
        """Returns the local path to the cached PDF, downloading it first if
        needed. Returns None (never raises) on any failure — callers treat a
        failed download as "this paper just can't be indexed right now"."""
        dest = self._cache_dir / f"{paper_id}.pdf"
        if dest.exists() and dest.stat().st_size > 0:
            return dest

        try:
            url_info = get_paper_pdf_url(paper_id)
            pdf_url = url_info.get("pdf_url")
        except Exception as e:
            logger.warning(f"[pdf_rag] could not resolve pdf url for {paper_id}: {e}")
            return None
        if not pdf_url:
            return None

        _pdf_rate_limiter.wait()
        tmp = dest.with_suffix(".part")
        try:
            with _session.get(pdf_url, stream=True, timeout=_DOWNLOAD_TIMEOUT) as resp:
                resp.raise_for_status()
                written = 0
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        if not chunk:
                            continue
                        written += len(chunk)
                        if written > _MAX_PDF_BYTES:
                            raise ValueError(f"PDF for {paper_id} exceeds {_MAX_PDF_BYTES} byte cap")
                        f.write(chunk)
            tmp.rename(dest)
            return dest
        except Exception as e:
            logger.warning(f"[pdf_rag] download failed for {paper_id}: {e}")
            tmp.unlink(missing_ok=True)
            return None


# ---------------------------------------------------------------------
# Extract + chunk
# ---------------------------------------------------------------------
class PdfChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def extract_and_chunk(
        self, path: Path, paper_id: str, extra_metadata: Dict[str, Any]
    ) -> List[Document]:
        try:
            loader = PyPDFLoader(str(path))
            loaded_docs = loader.load()
        except Exception as e:
            logger.warning(f"[pdf_rag] could not open PDF for {paper_id}: {e}")
            return []

        pages: List[str] = [doc.page_content or "" for doc in loaded_docs]

        full_text = "\n\n".join(t for t in pages if t.strip())
        if not full_text.strip():
            # Most likely a scanned PDF with no text layer. Fail soft — the
            # caller falls back to abstract-only context rather than crashing.
            logger.warning(f"[pdf_rag] no extractable text for {paper_id} (scanned PDF?)")
            return []

        chunks = self._splitter.split_text(full_text)
        return [
            Document(
                page_content=chunk,
                metadata={
                    "paper_id": paper_id,
                    "chunk_index": i,
                    "source_tool": "pdf_fulltext",
                    **extra_metadata,
                },
            )
            for i, chunk in enumerate(chunks)
        ]


# ---------------------------------------------------------------------
# Registry — what's already indexed (backs the "all" mode)
# ---------------------------------------------------------------------
class PaperIndexRegistry:
    def __init__(self, path: str = "./vector_store/registry.json"):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self._path.exists():
            self._write({})

    def _read(self) -> Dict[str, Any]:
        try:
            return json.loads(self._path.read_text())
        except Exception:
            return {}

    def _write(self, data: Dict[str, Any]) -> None:
        self._path.write_text(json.dumps(data, indent=2))

    def has(self, paper_id: str) -> bool:
        with self._lock:
            return paper_id in self._read()

    def set(self, paper_id: str, meta: Dict[str, Any]) -> None:
        with self._lock:
            data = self._read()
            data[paper_id] = meta
            self._write(data)

    def all_ids(self) -> List[str]:
        with self._lock:
            return list(self._read().keys())


# ---------------------------------------------------------------------
# Per-paper FAISS store
# ---------------------------------------------------------------------
class PdfVectorStore:
    def __init__(
        self,
        embeddings: Embeddings,
        registry: PaperIndexRegistry,
        store_dir: str = "./vector_store",
        max_loaded: int = 8,
    ):
        self._embeddings = embeddings
        self._registry = registry
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._loaded: "OrderedDict[str, FAISS]" = OrderedDict()
        self._max_loaded = max_loaded
        self._cache_lock = threading.Lock()

    def _index_dir(self, paper_id: str) -> Path:
        return self._store_dir / paper_id

    def has_persisted_index(self, paper_id: str) -> bool:
        return self._registry.has(paper_id) and (self._index_dir(paper_id) / "index.faiss").exists()

    def build(self, paper_id: str, docs: List[Document]) -> bool:
        if not docs:
            return False
        try:
            store = FAISS.from_documents(docs, self._embeddings)
            store.save_local(str(self._index_dir(paper_id)))
        except Exception as e:
            logger.warning(f"[pdf_rag] failed to build FAISS index for {paper_id}: {e}")
            return False
        self._registry.set(paper_id, {"chunks": len(docs), "title": docs[0].metadata.get("title")})
        self._cache_put(paper_id, store)
        return True

    def _cache_put(self, paper_id: str, store: FAISS) -> None:
        with self._cache_lock:
            self._loaded[paper_id] = store
            self._loaded.move_to_end(paper_id)
            while len(self._loaded) > self._max_loaded:
                self._loaded.popitem(last=False)

    def _load(self, paper_id: str) -> Optional[FAISS]:
        with self._cache_lock:
            if paper_id in self._loaded:
                self._loaded.move_to_end(paper_id)
                return self._loaded[paper_id]
        if not self.has_persisted_index(paper_id):
            return None
        try:
            store = FAISS.load_local(
                str(self._index_dir(paper_id)),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            logger.warning(f"[pdf_rag] failed to load FAISS index for {paper_id}: {e}")
            return None
        self._cache_put(paper_id, store)
        return store

    def search(self, paper_ids: List[str], query: str, k: int = 6) -> List[Document]:
        scored: List[Tuple[Document, float]] = []
        for paper_id in paper_ids:
            store = self._load(paper_id)
            if store is None:
                continue
            try:
                scored.extend(store.similarity_search_with_score(query, k=k))
            except Exception as e:
                logger.warning(f"[pdf_rag] search failed for {paper_id}: {e}")
        # FAISS's default index (IndexFlatL2) scores by L2 distance — lower is
        # more similar — so ascending sort gives the best matches first, even
        # when merging results pulled from several papers' indices.
        scored.sort(key=lambda pair: pair[1])
        return [doc for doc, _ in scored[:k]]


# ---------------------------------------------------------------------
# Orchestrator — this is what main.py talks to
# ---------------------------------------------------------------------
class PdfRagService:
    DEFAULT_K = 6

    def __init__(
        self,
        embeddings: Embeddings,
        cache_dir: str = "./pdf_cache",
        store_dir: str = "./vector_store",
    ):
        self._downloader = PdfDownloader(cache_dir)
        self._chunker = PdfChunker()
        self._registry = PaperIndexRegistry(f"{store_dir}/registry.json")
        self._store = PdfVectorStore(embeddings, self._registry, store_dir)
        self._paper_locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, paper_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._paper_locks.setdefault(paper_id, threading.Lock())

    def ensure_indexed(self, paper_id: str, on_status: StatusCallback = None) -> bool:
        """Idempotent: downloads + chunks + embeds + persists a paper's FAISS
        index if it isn't already there. Returns whether the paper is now
        (or already was) searchable."""
        if self._store.has_persisted_index(paper_id):
            return True
        with self._lock_for(paper_id):
            if self._store.has_persisted_index(paper_id):  # re-check post-lock
                return True
            if on_status:
                on_status("pdf_rag", f"Downloading and indexing {paper_id}…")

            path = self._downloader.download(paper_id)
            if path is None:
                return False

            title, authors, url = paper_id, "", f"https://arxiv.org/abs/{paper_id}"
            try:
                details = get_paper_details(paper_id)
                title = details.get("title") or title
                authors = ", ".join(details.get("authors", []) or [])
                url = details.get("abstract_url") or url
            except Exception:
                pass  # best-effort citation enrichment only — never block indexing on this

            docs = self._chunker.extract_and_chunk(
                path, paper_id, extra_metadata={"title": title, "authors": authors, "url": url}
            )
            return self._store.build(paper_id, docs)

    def retrieve(
        self,
        paper_ids: List[str],
        query: str,
        k: Optional[int] = None,
        on_status: StatusCallback = None,
    ) -> List[Document]:
        """paper_ids may be explicit arXiv ids, or ["all"] to search every
        paper already indexed this session. Returns [] (never raises) when
        nothing is indexed yet or every requested paper fails to download —
        the graph's existing no_results node handles that gracefully."""
        k = k or self.DEFAULT_K
        wants_all = any(str(p).strip().lower() == "all" for p in paper_ids)

        if wants_all:
            targets = self._registry.all_ids()
        else:
            targets = [pid for pid in paper_ids if self.ensure_indexed(pid, on_status=on_status)]

        if not targets:
            return []
        return self._store.search(targets, query, k=k)