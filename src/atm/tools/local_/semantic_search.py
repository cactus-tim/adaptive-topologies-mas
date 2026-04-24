"""SemanticSearchTool — numpy TF-IDF cosine similarity search over a corpus.

Public API
----------
SemanticSearchTool -- Tool implementation using numpy TF-IDF with cosine similarity
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from atm.core.types import ToolResult
from atm.tools.base import ToolSchema

# Snippet preview length (first N characters of document content)
_SNIPPET_LEN = 120


def _tokenize(text: str) -> list[str]:
    """Simple whitespace+punctuation tokenizer returning lowercased tokens."""
    import re

    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _build_tfidf(
    docs: list[list[str]],
) -> tuple[dict[str, int], np.ndarray]:
    """Build TF-IDF matrix for a list of tokenized documents.

    Uses sklearn-style smoothed IDF:
        idf = log((1 + N) / (1 + df)) + 1

    Documents are L2-normalised so cosine similarity reduces to a dot product.

    Parameters
    ----------
    docs:
        List of token lists (one per document).

    Returns
    -------
    vocab:
        Mapping of token -> column index.
    tfidf_matrix:
        Shape (N, V) float64 matrix, L2-normalised rows.
    """
    # Build vocabulary
    vocab: dict[str, int] = {}
    for tokens in docs:
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)

    n_docs = len(docs)
    n_terms = len(vocab)

    if n_terms == 0 or n_docs == 0:
        return vocab, np.zeros((n_docs, 0), dtype=np.float64)

    # Compute TF (term frequency, raw counts / doc length)
    tf = np.zeros((n_docs, n_terms), dtype=np.float64)
    for i, tokens in enumerate(docs):
        if not tokens:
            continue
        for t in tokens:
            j = vocab[t]
            tf[i, j] += 1.0
        tf[i] /= len(tokens)

    # Compute IDF: log((1 + N) / (1 + df)) + 1  (sklearn smooth)
    df = np.zeros(n_terms, dtype=np.float64)
    for tokens in docs:
        for t in set(tokens):
            j = vocab[t]
            df[j] += 1.0

    idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0

    # TF-IDF
    tfidf = tf * idf

    # L2-normalise each row
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tfidf /= norms

    return vocab, tfidf


class SemanticSearchTool:
    """Search a corpus of text documents using TF-IDF cosine similarity.

    The corpus is loaded from ``corpus_dir`` at construction time.
    Each ``.txt`` file in the directory is treated as one document.
    Document IDs are the file stems (without the ``.txt`` extension).

    Parameters
    ----------
    corpus_dir:
        Directory containing ``*.txt`` corpus files.
    top_k:
        Maximum number of hits to return (default 3).

    Output shape
    ------------
    hits : list of {doc_id: str, score: float, snippet: str}

    Error cases
    -----------
    Empty or whitespace-only query → ok=False, error="empty query".
    """

    name: ClassVar[str] = "semantic_search"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="semantic_search",
        description=(
            "Search a text corpus using TF-IDF cosine similarity. "
            "Returns ranked hits with doc_id, similarity score, and a snippet."
        ),
        parameters={
            "query": {
                "type": "string",
                "description": "Search query string.",
            },
        },
        returns={
            "hits": "array of {doc_id: string, score: float, snippet: string}",
        },
    )

    def __init__(self, corpus_dir: Path, top_k: int = 3) -> None:
        self._corpus_dir = corpus_dir
        self._top_k = top_k
        self._doc_ids: list[str] = []
        self._snippets: list[str] = []
        self._vocab: dict[str, int] = {}
        self._tfidf_matrix: np.ndarray = np.zeros((0, 0))
        self._idf: np.ndarray = np.zeros(0)
        self._loaded = False

    def _load_corpus(self) -> None:
        """Load corpus documents and build TF-IDF index."""
        txt_files = sorted(self._corpus_dir.glob("*.txt"))
        raw_texts: list[str] = []
        doc_ids: list[str] = []

        for f in txt_files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            doc_ids.append(f.stem)
            raw_texts.append(text)

        self._doc_ids = doc_ids
        self._snippets = [t[:_SNIPPET_LEN] for t in raw_texts]

        tokenized_docs = [_tokenize(t) for t in raw_texts]

        # Store IDF vector for query vectorisation
        n_docs = len(tokenized_docs)
        vocab: dict[str, int] = {}
        for tokens in tokenized_docs:
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)

        n_terms = len(vocab)
        df = np.zeros(n_terms, dtype=np.float64)
        for tokens in tokenized_docs:
            for t in set(tokens):
                j = vocab[t]
                df[j] += 1.0

        idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0

        self._vocab = vocab
        self._idf = idf
        self._vocab_size = n_terms

        self._vocab, self._tfidf_matrix = _build_tfidf(tokenized_docs)
        self._loaded = True

    def _vectorize_query(self, tokens: list[str]) -> np.ndarray:
        """Convert query tokens into a normalised TF-IDF vector."""
        n_terms = len(self._vocab)
        if n_terms == 0:
            return np.zeros(0)

        vec: np.ndarray = np.zeros(n_terms, dtype=np.float64)
        for t in tokens:
            j = self._vocab.get(t)
            if j is not None:
                vec[j] += 1.0

        if vec.sum() == 0:
            return vec

        # Apply IDF
        vec = vec * self._idf

        # L2-normalise
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm

        return vec

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        """Search the corpus for the given query.

        Parameters
        ----------
        args:
            query : str — the search query

        Returns
        -------
        ToolResult
            ok=True with hits on success.
            ok=False with error="empty query" for empty/whitespace queries.
        """
        call_id = uuid.uuid4()
        query: str = args.get("query", "")

        if not query or not query.strip():
            return ToolResult(
                call_id=call_id,
                ok=False,
                output=None,
                error="empty query",
                latency_ms=0,
            )

        if not self._loaded:
            self._load_corpus()

        query_tokens = _tokenize(query)
        query_vec = self._vectorize_query(query_tokens)

        n_docs = len(self._doc_ids)

        if n_docs == 0 or self._tfidf_matrix.shape[1] == 0 or query_vec.sum() == 0:
            return ToolResult(
                call_id=call_id,
                ok=True,
                output={"hits": []},
                error=None,
                latency_ms=0,
            )

        # Cosine similarity (both vectors are L2-normalised → dot product)
        scores: np.ndarray = self._tfidf_matrix @ query_vec

        # Stable argsort descending
        ranked_indices = np.argsort(-scores, kind="stable")

        hits: list[dict[str, Any]] = []
        for idx in ranked_indices[: self._top_k]:
            score = float(scores[idx])
            hits.append(
                {
                    "doc_id": self._doc_ids[idx],
                    "score": score,
                    "snippet": self._snippets[idx],
                }
            )

        return ToolResult(
            call_id=call_id,
            ok=True,
            output={"hits": hits},
            error=None,
            latency_ms=0,
        )


__all__ = ["SemanticSearchTool"]
