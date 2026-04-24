"""SemanticSearchTool — numpy TF-IDF vector search over a local corpus directory."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from atm.core.types import ToolResult
from atm.tools.base import ToolSchema


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters, filtering empty tokens."""
    return [tok for tok in re.split(r"[^a-z0-9]+", text.lower()) if tok]


class SemanticSearchTool:
    """Corpus search tool using numpy TF-IDF with cosine similarity.

    The index is built once at construction time from all ``.txt`` files in
    ``corpus_dir``.  At query time a TF-IDF vector is computed for the query
    and compared against the pre-computed (L2-normalised) document vectors.

    Args:
        corpus_dir: Directory containing ``.txt`` corpus documents.
        top_k:      Maximum number of hits to return (default 5).
    """

    name: ClassVar[str] = "semantic_search"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="semantic_search",
        description=(
            "Full-text semantic search over a local corpus using TF-IDF. "
            "Returns ranked document snippets that are most similar to the query."
        ),
        parameters={
            "query": "string — the search query",
            "k": "integer (optional) — override top_k for this call",
        },
        returns={
            "hits": "array of {doc_id: string, score: float, snippet: string}",
        },
    )

    def __init__(self, corpus_dir: Path, top_k: int = 5) -> None:
        self.corpus_dir = corpus_dir
        self.top_k = top_k
        self._build_index()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Load corpus, build vocab, compute TF-IDF matrix and L2-normalise."""
        txt_files = sorted(self.corpus_dir.glob("*.txt"))
        self._doc_ids: list[str] = []
        self._doc_texts: list[str] = []
        tokenized_docs: list[list[str]] = []

        for path in txt_files:
            text = path.read_text(encoding="utf-8")
            self._doc_ids.append(path.stem)
            self._doc_texts.append(text)
            tokenized_docs.append(_tokenize(text))

        n_docs = len(tokenized_docs)

        # Build vocabulary: term → column index (sorted for determinism)
        vocab: dict[str, int] = {}
        for tokens in tokenized_docs:
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        self._vocab = vocab
        n_terms = len(vocab)

        if n_docs == 0 or n_terms == 0:
            self._tfidf_matrix = np.zeros((n_docs, n_terms), dtype=np.float64)
            return

        # TF matrix: raw term counts per document, then normalise by doc length
        tf = np.zeros((n_docs, n_terms), dtype=np.float64)
        for doc_idx, tokens in enumerate(tokenized_docs):
            for tok in tokens:
                tf[doc_idx, vocab[tok]] += 1.0

        # Normalise TF by total term count in each document (term frequency)
        row_sums = tf.sum(axis=1, keepdims=True)
        # Avoid division by zero for empty docs
        row_sums[row_sums == 0] = 1.0
        tf /= row_sums

        # IDF: sklearn-smoothed formula  idf[t] = log((1 + N) / (1 + df[t])) + 1
        df = (tf > 0).sum(axis=0).astype(np.float64)  # document frequency per term
        idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0

        # TF-IDF and L2 normalisation
        tfidf = tf * idf  # broadcast: (n_docs, n_terms) * (n_terms,)
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._tfidf_matrix = tfidf / norms  # L2-normalised row vectors

        # Store IDF for query vectorisation
        self._idf = idf

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _vectorize_query(self, tokens: list[str]) -> np.ndarray:
        """Build a TF-IDF vector for the query using the corpus vocab/IDF."""
        n_terms = len(self._vocab)
        vec = np.zeros(n_terms, dtype=np.float64)
        for tok in tokens:
            if tok in self._vocab:
                vec[self._vocab[tok]] += 1.0

        total = vec.sum()
        if total > 0:
            vec /= total  # TF normalisation

        vec *= self._idf  # apply IDF

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm  # L2 normalise

        return vec

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        """Search the corpus.

        Args:
            args: dict with keys:
                ``query`` (str) — the search query (required).
                ``k`` (int, optional) — override top_k for this call.

        Returns:
            ToolResult with ``output["hits"]`` as a list of
            ``{"doc_id": str, "score": float, "snippet": str}`` dicts,
            ordered by descending cosine similarity.

            If the query is empty or all-whitespace: ``ok=False``,
            ``error="empty query"``.

            If the query has no vocabulary overlap with the corpus:
            ``ok=True``, ``hits=[]``.
        """
        raw_query: str = args.get("query", "")
        k: int = int(args.get("k") or self.top_k)

        if not raw_query.strip():
            return ToolResult(
                call_id=uuid.uuid4(),
                ok=False,
                output=None,
                error="empty query",
                latency_ms=0,
            )

        tokens = _tokenize(raw_query)
        query_vec = self._vectorize_query(tokens)

        # All-zero vector means no vocab overlap
        if np.all(query_vec == 0.0):
            return ToolResult(
                call_id=uuid.uuid4(),
                ok=True,
                output={"hits": []},
                error=None,
                latency_ms=0,
            )

        # Cosine similarity: dot product (both sides already L2-normalised)
        scores: np.ndarray = self._tfidf_matrix.dot(query_vec)

        # Top-k by descending score (stable sort for deterministic tie-breaking)
        top_indices = np.argsort(-scores, kind="stable")[:k]

        hits: list[dict[str, Any]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                # Skip non-matching documents
                continue
            snippet = self._doc_texts[idx][:200]
            hits.append(
                {
                    "doc_id": self._doc_ids[idx],
                    "score": score,
                    "snippet": snippet,
                }
            )

        return ToolResult(
            call_id=uuid.uuid4(),
            ok=True,
            output={"hits": hits},
            error=None,
            latency_ms=0,
        )
