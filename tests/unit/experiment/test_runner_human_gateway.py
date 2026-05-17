"""Unit tests for _build_human_gateway factory in runner.py (M14 Step 5.1).

Four scenarios:
  1. human_cfg=None     → (None, None)
  2. enabled=False      → (None, None)
  3. gateway="llm_simulated" → (None, LLMWrapper)
  4. gateway="streamlit"     → (StreamlitHumanGateway, LLMWrapper)

No live DB or real LLM required — all external collaborators are mocked.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from atm.experiment.config import HumanCfg
from atm.experiment.runner import _build_human_gateway
from atm.human.streamlit_gateway import StreamlitHumanGateway
from atm.llm.wrapper import LLMWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_llm_wrapper() -> LLMWrapper:
    """Return a MagicMock that satisfies isinstance(..., LLMWrapper) checks."""
    wrapper = MagicMock(spec=LLMWrapper)
    return wrapper


def _make_fake_pricing() -> Any:
    return MagicMock()


def _make_fake_budget() -> Any:
    return MagicMock()


# ---------------------------------------------------------------------------
# Scenario 1: human_cfg=None → (None, None)
# ---------------------------------------------------------------------------


def test_build_human_gateway_none_cfg_returns_none_none() -> None:
    """When human_cfg is None, factory returns (None, None)."""
    fake_pricing = _make_fake_pricing()
    fake_budget = _make_fake_budget()

    primary, fallback = _build_human_gateway(
        human_cfg=None,
        pricing=fake_pricing,
        budget=fake_budget,
        default_model_id="fake:echo",
        fake_fixtures={},
    )

    assert primary is None
    assert fallback is None


# ---------------------------------------------------------------------------
# Scenario 2: enabled=False → (None, None)
# ---------------------------------------------------------------------------


def test_build_human_gateway_disabled_returns_none_none() -> None:
    """When human_cfg.enabled=False, factory returns (None, None)."""
    human_cfg = HumanCfg(enabled=False, gateway="llm_simulated")
    fake_pricing = _make_fake_pricing()
    fake_budget = _make_fake_budget()

    primary, fallback = _build_human_gateway(
        human_cfg=human_cfg,
        pricing=fake_pricing,
        budget=fake_budget,
        default_model_id="fake:echo",
        fake_fixtures={},
    )

    assert primary is None
    assert fallback is None


# ---------------------------------------------------------------------------
# Scenario 3: gateway="llm_simulated" → (None, LLMWrapper)
# ---------------------------------------------------------------------------


def test_build_human_gateway_llm_simulated_returns_none_and_wrapper() -> None:
    """gateway='llm_simulated' returns (None, LLMWrapper) — no primary gateway."""
    human_cfg = HumanCfg(enabled=True, gateway="llm_simulated", model="fake:echo")
    fake_pricing = _make_fake_pricing()
    fake_budget = _make_fake_budget()

    fake_wrapper = _make_fake_llm_wrapper()

    with patch("atm.experiment.runner.build_llm", return_value=fake_wrapper) as mock_build_llm:
        primary, fallback = _build_human_gateway(
            human_cfg=human_cfg,
            pricing=fake_pricing,
            budget=fake_budget,
            default_model_id="fake:echo",
            fake_fixtures={},
        )

    assert primary is None
    assert fallback is fake_wrapper
    mock_build_llm.assert_called_once()


# ---------------------------------------------------------------------------
# Scenario 4: gateway="streamlit" → (StreamlitHumanGateway, LLMWrapper)
# ---------------------------------------------------------------------------


def test_build_human_gateway_streamlit_returns_gateway_and_wrapper() -> None:
    """gateway='streamlit' builds a StreamlitHumanGateway and a fallback LLMWrapper.

    - HumanRequestQueue is constructed from a mocked session factory.
    - StreamlitHumanGateway wraps the queue.
    - Fallback LLMWrapper comes from fallback_llm_model or default_model_id.
    - No live DB required; session factory is mocked.
    """
    human_cfg = HumanCfg(
        enabled=True,
        gateway="streamlit",
        queue_dsn="postgresql+asyncpg://atm:atm@localhost/testdb",
        participant_id="participant-42",
        study_session_id="session-abc",
        fallback_llm_model="fake:echo",
    )
    fake_pricing = _make_fake_pricing()
    fake_budget = _make_fake_budget()

    fake_wrapper = _make_fake_llm_wrapper()
    fake_session_factory = MagicMock()
    fake_queue = MagicMock()

    with (
        patch("atm.experiment.runner.build_llm", return_value=fake_wrapper) as mock_build_llm,
        patch(
            "atm.experiment.runner.create_engine", return_value=MagicMock()
        ) as mock_create_engine,
        patch(
            "atm.experiment.runner.create_session_factory",
            return_value=fake_session_factory,
        ) as mock_create_sf,
        # HumanRequestQueue is lazily imported inside _build_human_gateway;
        # patch the source module so the lazy import resolves to our mock.
        patch(
            "atm.human._queue.HumanRequestQueue",
            return_value=fake_queue,
        ) as mock_hrq,
    ):
        primary, fallback = _build_human_gateway(
            human_cfg=human_cfg,
            pricing=fake_pricing,
            budget=fake_budget,
            default_model_id="fake:echo",
            fake_fixtures={},
        )

    # Primary must be a StreamlitHumanGateway wrapping our fake queue
    assert isinstance(primary, StreamlitHumanGateway), (
        f"Expected StreamlitHumanGateway, got {type(primary)}"
    )

    # Fallback LLM wrapper must be returned
    assert fallback is fake_wrapper

    # Verify that create_engine was called with the queue_dsn
    mock_create_engine.assert_called_once()
    engine_call_args = mock_create_engine.call_args
    assert "postgresql+asyncpg://atm:atm@localhost/testdb" in str(engine_call_args)

    # Verify session factory was created from the engine
    mock_create_sf.assert_called_once()

    # Verify HumanRequestQueue was instantiated with the session factory
    mock_hrq.assert_called_once_with(session_factory=fake_session_factory)

    # build_llm must be called at least once (for fallback)
    mock_build_llm.assert_called_once()
