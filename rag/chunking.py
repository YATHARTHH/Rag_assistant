import math
import re


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2, strict=False))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 * norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def split_into_sentences(text: str) -> list[str]:
    """
    Splits text into clean individual sentences using regex lookbehinds.
    Protects numbered list digits and abbreviations.
    """
    text = re.sub(r"\s+", " ", text).strip()
    sentence_boundary = re.compile(
        r"(?<!\b[0-9])"  # No digit before period (e.g., 1. )
        r"(?<!\b[A-Za-z])"  # No single letter before period (e.g., A. )
        r"(?<!\b[eE]\.[gG])"  # No e.g.
        r"(?<!\b[iI]\.[eE])"  # No i.e.
        r"(?<!\b[vV][sS])"  # No vs.
        r"(?<=[.!?])\s+"
    )
    sentences = sentence_boundary.split(text)
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunk_text(content: str, title: str, embedder, distance_threshold=0.5) -> list[dict]:
    """
    Groups document content into semantic chunks based on sentence distance boundaries.
    Falls back to paragraph-based splitting if natural paragraph breaks exist.
    """
    # Split by natural paragraph boundaries (double newlines)
    paragraphs = [p.strip() for p in re.split(r"\r?\n\s*\r?\n", content) if p.strip()]
    if len(paragraphs) > 1:
        chunks = []
        for idx, p in enumerate(paragraphs):
            chunks.append(
                {"title": title, "content": re.sub(r"\s+", " ", p).strip(), "sent_index": idx}
            )
        return chunks

    sentences = split_into_sentences(content)
    if len(sentences) <= 1:
        return [
            {"title": title, "content": s, "sent_index": idx} for idx, s in enumerate(sentences)
        ]

    # Generate embeddings for all sentences
    embeddings = embedder.embed_documents(sentences)

    # Calculate cosine distances between adjacent sentences
    distances = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        distances.append(1.0 - sim)

    # Custom semantic chunk clustering
    chunks = []
    current_chunk_sentences = [sentences[0]]
    chunk_index = 0

    for i, dist in enumerate(distances):
        # If semantic gap exceeds threshold, start a new chunk
        if dist >= distance_threshold:
            chunks.append(
                {
                    "title": title,
                    "content": " ".join(current_chunk_sentences),
                    "sent_index": chunk_index,
                }
            )
            current_chunk_sentences = []
            chunk_index += 1
        current_chunk_sentences.append(sentences[i + 1])

    if current_chunk_sentences:
        chunks.append(
            {
                "title": title,
                "content": " ".join(current_chunk_sentences),
                "sent_index": chunk_index,
            }
        )
    return chunks


def parent_child_chunking(content: str, filename: str, embedder) -> list[dict]:
    """
    Splits document content into semantic parents (paragraphs) and child chunks (sentences).
    """
    parents = semantic_chunk_text(content, filename, embedder)
    chunks = []
    for _parent_idx, parent in enumerate(parents):
        parent_text = parent["content"]
        sentences = split_into_sentences(parent_text)
        overlap_size = 1
        for i, sent in enumerate(sentences):
            start = max(0, i - overlap_size)
            end = min(len(sentences), i + overlap_size + 1)
            window_text = " ".join(sentences[start:end])
            chunks.append(
                {
                    "title": filename,
                    "content": sent,
                    "parent_text": parent_text,
                    "overlap_text": window_text,
                    "sent_index": len(chunks),
                }
            )
    return chunks


def chunk_document_text(
    content: str, title: str, chunk_size=None, chunk_overlap=None
) -> list[dict]:
    """
    Backup chunker (used if semantic embeddings are skipped).
    """
    sentences = split_into_sentences(content)
    return [
        {"title": title, "content": sent, "sent_index": idx} for idx, sent in enumerate(sentences)
    ]
