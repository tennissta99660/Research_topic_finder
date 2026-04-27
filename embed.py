# embed.py — SBERT embedding + cosine similarity for CARTOGRAPH
import logging

import numpy as np

from config import EMBED_MODEL
from db import get_unembedded_papers, update_embedding

logger = logging.getLogger(__name__)

# Module-level singleton to avoid re-downloading model every call
_model = None


def _get_model():
    """Lazy-load the SBERT model (singleton). Import is deferred to avoid slow startup."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading SBERT model: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr, b_arr = np.array(a), np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-8))


def embed_papers(topic_id: int) -> int:
    """Compute SBERT embeddings for all unembedded papers in a topic.

    Embeds in batch for efficiency. Returns the number of papers embedded.
    """
    model = _get_model()
    papers = get_unembedded_papers(topic_id)

    if not papers:
        logger.info("No unembedded papers found.")
        return 0

    # Build text inputs: title + abstract
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]

    logger.info(f"Embedding {len(texts)} papers in batch...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    for paper, emb in zip(papers, embeddings):
        update_embedding(paper["id"], emb.tolist())

    logger.info(f"Embedded {len(papers)} papers")
    return len(papers)
