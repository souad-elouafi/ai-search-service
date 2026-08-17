import os
import threading
from typing import Sequence

from app.config import (
    AI_EMBEDDING_BATCH_SIZE,
    AI_TORCH_INTEROP_THREADS,
    AI_TORCH_NUM_THREADS,
)

# These defaults are set before importing torch/sentence-transformers. Operators
# can still override any of them explicitly in the process environment.
os.environ.setdefault("OMP_NUM_THREADS", str(AI_TORCH_NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(AI_TORCH_NUM_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(AI_TORCH_NUM_THREADS))

import torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(AI_TORCH_NUM_THREADS)
try:
    torch.set_num_interop_threads(AI_TORCH_INTEROP_THREADS)
except RuntimeError:
    # PyTorch only permits this setting before inter-op work starts. This can
    # occur under reloaders/tests that imported torch before this module.
    pass

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model = None
_model_lock = threading.Lock()


def get_model():
    """Create the embedding model at most once in this process."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(MODEL_NAME)
    return _model


def generate_embeddings(texts: Sequence[str]):
    """Encode texts in bounded batches while preserving normalized vectors."""
    if not texts:
        return []
    values = texts if isinstance(texts, list) else list(texts)
    with torch.inference_mode():
        return get_model().encode(
            values,
            batch_size=AI_EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


def generate_embedding(text: str):
    return generate_embeddings([text])[0]
