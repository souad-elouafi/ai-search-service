"""Synthetic embedding/index benchmark; never contacts ChedMed."""
import argparse
from pathlib import Path
import sys
import time
import math

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import embedding_service, faiss_service


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", type=int, default=100)
    parser.add_argument(
        "--synthetic-model",
        action="store_true",
        help="Use a deterministic local model stub to measure call counts without a model download.",
    )
    args = parser.parse_args()
    products = [
        {"id": str(i), "name": f"Produit {i}", "description": "description synthetic", "category": "test"}
        for i in range(args.products)
    ]

    model = None
    if args.synthetic_model:
        class SyntheticModel:
            encode_calls = 0

            def encode(self, texts, **kwargs):
                self.encode_calls += 1
                if isinstance(texts, str):
                    texts = [texts]
                vectors = np.zeros((len(texts), faiss_service.DIMENSION), dtype="float32")
                vectors[:, 0] = 1.0
                return vectors

        model = SyntheticModel()
        embedding_service._model = model

    start = time.perf_counter()
    embedding_service.generate_embedding("un produit")
    single_time = time.perf_counter() - start
    start = time.perf_counter()
    embedding_service.generate_embeddings([faiss_service._product_text(p) for p in products])
    batch_time = time.perf_counter() - start
    start = time.perf_counter()
    faiss_service.replace_products(products)
    rebuild_time = time.perf_counter() - start
    output = {
        "products": args.products,
        "single_embedding_seconds": round(single_time, 4),
        "batch_embedding_seconds": round(batch_time, 4),
        "batch_rebuild_seconds": round(rebuild_time, 4),
        "optimized_encode_calls_for_rebuild": math.ceil(
            args.products / faiss_service.AI_INDEX_BUILD_BATCH_SIZE
        ),
        "legacy_encode_calls_for_rebuild": args.products,
    }
    if model is not None:
        output["total_measured_encode_calls"] = model.encode_calls
        output["timings_use_synthetic_model"] = True
    print(output)


if __name__ == "__main__":
    main()
