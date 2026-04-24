"""Unit tests for SemanticSearchTool (numpy TF-IDF)."""

from __future__ import annotations

import pathlib

import pytest

from atm.tools.local_.semantic_search import SemanticSearchTool

CORPUS_DIR = pathlib.Path(__file__).parent.parent.parent / "fixtures" / "tools" / "corpus"


@pytest.fixture()
def tool() -> SemanticSearchTool:
    """Default SemanticSearchTool with top_k=5."""
    return SemanticSearchTool(corpus_dir=CORPUS_DIR, top_k=5)


@pytest.mark.asyncio
async def test_empty_query(tool: SemanticSearchTool) -> None:
    result = await tool.ainvoke({"query": ""})
    assert result.ok is False
    assert result.error == "empty query"


@pytest.mark.asyncio
async def test_whitespace_query(tool: SemanticSearchTool) -> None:
    result = await tool.ainvoke({"query": "   "})
    assert result.ok is False
    assert result.error == "empty query"


@pytest.mark.asyncio
async def test_neural_query_returns_doc1_top(tool: SemanticSearchTool) -> None:
    result = await tool.ainvoke({"query": "neural network"})
    assert result.ok is True
    hits = result.output["hits"]
    assert len(hits) > 0
    assert hits[0]["doc_id"] == "doc_1"
    assert hits[0]["score"] > 0.0
    assert isinstance(hits[0]["snippet"], str)
    assert len(hits[0]["snippet"]) <= 200


@pytest.mark.asyncio
async def test_carrot_query_returns_doc2_top(tool: SemanticSearchTool) -> None:
    result = await tool.ainvoke({"query": "carrot cake"})
    assert result.ok is True
    hits = result.output["hits"]
    assert len(hits) > 0
    assert hits[0]["doc_id"] == "doc_2"


@pytest.mark.asyncio
async def test_paris_query_returns_doc3_top(tool: SemanticSearchTool) -> None:
    result = await tool.ainvoke({"query": "capital of France Paris"})
    assert result.ok is True
    hits = result.output["hits"]
    assert len(hits) > 0
    assert hits[0]["doc_id"] == "doc_3"


@pytest.mark.asyncio
async def test_unknown_query_returns_empty_hits(tool: SemanticSearchTool) -> None:
    result = await tool.ainvoke({"query": "xyzzy_nonsense"})
    assert result.ok is True
    hits = result.output["hits"]
    assert hits == []


@pytest.mark.asyncio
async def test_top_k_limits_results() -> None:
    tool = SemanticSearchTool(corpus_dir=CORPUS_DIR, top_k=1)
    result = await tool.ainvoke({"query": "neural"})
    assert result.ok is True
    hits = result.output["hits"]
    assert len(hits) == 1


@pytest.mark.asyncio
async def test_top_k_override_via_args(tool: SemanticSearchTool) -> None:
    """k passed in args overrides top_k default."""
    result = await tool.ainvoke({"query": "the", "k": 2})
    assert result.ok is True
    hits = result.output["hits"]
    assert len(hits) <= 2


@pytest.mark.asyncio
async def test_hit_shape(tool: SemanticSearchTool) -> None:
    """Each hit must have doc_id, score, snippet keys."""
    result = await tool.ainvoke({"query": "neural"})
    assert result.ok is True
    for hit in result.output["hits"]:
        assert "doc_id" in hit
        assert "score" in hit
        assert "snippet" in hit
        assert isinstance(hit["score"], float)
