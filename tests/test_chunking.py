import pytest

from rag.chunking import (
    chunk_document_text,
    cosine_similarity,
    parent_child_chunking,
    semantic_chunk_text,
    split_into_sentences,
)


def test_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]

    assert cosine_similarity(v1, v2) == pytest.approx(1.0)
    assert cosine_similarity(v1, v3) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [0, 0]) == 0.0


def test_split_into_sentences():
    text = "First sentence. Second sentence! Is this third? Yes e.g. for testing."
    sentences = split_into_sentences(text)
    assert len(sentences) >= 3
    assert sentences[0] == "First sentence."
    assert sentences[1] == "Second sentence!"


def test_semantic_chunk_text_with_paragraphs(mock_embedder):
    content = "Paragraph 1 is here.\n\nParagraph 2 is here."
    chunks = semantic_chunk_text(content, "test.txt", mock_embedder)
    assert len(chunks) == 2
    assert chunks[0]["content"] == "Paragraph 1 is here."
    assert chunks[1]["content"] == "Paragraph 2 is here."


def test_semantic_chunk_text_single_sentence(mock_embedder):
    content = "Single short sentence."
    chunks = semantic_chunk_text(content, "test.txt", mock_embedder)
    assert len(chunks) == 1
    assert chunks[0]["content"] == "Single short sentence."


def test_parent_child_chunking(mock_embedder):
    content = "Sentence one. Sentence two."
    chunks = parent_child_chunking(content, "doc.pdf", mock_embedder)
    assert len(chunks) == 2
    assert "parent_text" in chunks[0]
    assert "overlap_text" in chunks[0]


def test_backup_chunk_document_text():
    content = "Hello world. Second line."
    chunks = chunk_document_text(content, "doc.txt")
    assert len(chunks) == 2
    assert chunks[0]["title"] == "doc.txt"
