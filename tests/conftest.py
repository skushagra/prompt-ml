"""
Shared fixtures and configuration for the prompt-ml test suite.

Integration tests (marked with @pytest.mark.integration) require
OPENAI_API_KEY to be set in the environment.  They are skipped
automatically when the key is absent.

Run only unit tests (MockBackend, no API calls):
    uv run --with pytest pytest tests/ -m "not integration" -v

Run only integration tests (real OpenAI):
    uv run --with pytest --with openai pytest tests/ -m integration -v

Run everything:
    uv run --with pytest --with openai pytest tests/ -v
"""

import os

import pytest

from prompt_ml.backend.openai_backend import OpenAIBackend


def _api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: calls the real OpenAI API — requires OPENAI_API_KEY",
    )


@pytest.fixture(scope="session")
def openai_backend():
    """
    A real OpenAIBackend shared across the whole integration test session.

    Skips the entire session if OPENAI_API_KEY is not set.
    Uses gpt-4o-mini for cost efficiency.
    """
    key = _api_key()
    if not key:
        pytest.skip(
            "OPENAI_API_KEY not set — skipping integration tests. "
            "Export it and re-run with: pytest tests/ -m integration"
        )
    return OpenAIBackend(api_key=key, model="gpt-4o-mini", temperature=0.0)
