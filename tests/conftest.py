import os
import sys
from pathlib import Path

import pytest

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Enforce mock environment variables before importing application modules
os.environ["GROQ_API_KEY"] = "gsk_mock_test_key_for_unit_tests"
os.environ["JWT_SECRET"] = "mock_secret_key_for_testing_purposes"
os.environ["RAG_API_KEY"] = "rag_developer_key_123"


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path):
    """Fixture to point SQLite user DB to a temporary directory for test isolation."""
    import database.sqlite as sqlite_mod

    original_db = sqlite_mod.USER_DB_PATH
    temp_db_path = str(tmp_path / "test_users.db")
    sqlite_mod.USER_DB_PATH = temp_db_path
    yield
    sqlite_mod.USER_DB_PATH = original_db


class MockEmbedder:
    """Mock Embedder for testing semantic chunking without downloading models."""

    def embed_documents(self, texts):
        # Return deterministic dummy 4-dimensional vectors based on text length
        return [[float(len(t)), float(i), 1.0, 0.5] for i, t in enumerate(texts)]


@pytest.fixture
def mock_embedder():
    return MockEmbedder()
