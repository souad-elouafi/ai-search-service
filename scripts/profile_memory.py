"""Development-only RSS profiler for model, FAISS, sync, and repeated operations."""
import argparse
import json
import os
import resource
import sys
import threading
import tracemalloc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def current_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        pass
    statm = Path("/proc/self/statm")
    if statm.exists():
        resident_pages = int(statm.read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1024 / 1024
    # Last-resort fallback is peak rather than current RSS on macOS.
    return peak_rss_mb()


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return value / (1024 if sys.platform.startswith("linux") else 1024 * 1024)


def checkpoint(name: str, **details):
    current, peak = tracemalloc.get_traced_memory()
    print(json.dumps({
        "checkpoint": name,
        "rss_mb": round(current_rss_mb(), 2),
        "peak_rss_mb": round(peak_rss_mb(), 2),
        "python_traced_mb": round(current / 1024 / 1024, 2),
        "python_traced_peak_mb": round(peak / 1024 / 1024, 2),
        **details,
    }, sort_keys=True))


def run_with_rss_peak(operation):
    samples = [current_rss_mb()]
    stopped = threading.Event()

    def sample():
        while not stopped.wait(0.005):
            samples.append(current_rss_mb())

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        result = operation()
    finally:
        stopped.set()
        sampler.join()
        samples.append(current_rss_mb())
    return result, max(samples)


def synthetic_products(count: int):
    return [
        {
            "id": str(i),
            "name": f"Produit {i}",
            "description": "Description catalogue synthetique pour profilage memoire",
            "category": f"categorie-{i % 20}",
            "price": i % 1000,
            "status": "ACTIVE",
        }
        for i in range(count)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", type=int, default=10_000)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--real-model", action="store_true")
    args = parser.parse_args()

    tracemalloc.start()
    checkpoint("process_start")

    global np
    import numpy as np
    from app.services import embedding_service, faiss_service
    from app.services import backup_sync_service

    checkpoint("after_imports", model_loaded=embedding_service._model is not None)

    if args.real_model:
        embedding_service.get_model()
        model_kind = "real"
    else:
        class SyntheticModel:
            encode_calls = 0

            def encode(self, texts, **kwargs):
                self.encode_calls += 1
                if isinstance(texts, str):
                    texts = [texts]
                vectors = np.zeros((len(texts), faiss_service.DIMENSION), dtype="float32")
                for row, text in enumerate(texts):
                    vectors[row, hash(text) % faiss_service.DIMENSION] = 1.0
                return vectors

        embedding_service._model = SyntheticModel()
        model_kind = "synthetic"
    checkpoint("after_model_load", model=model_kind)

    faiss_service._reset_state()
    checkpoint("after_empty_faiss")

    items = synthetic_products(args.products)
    checkpoint("after_loading_products", products=args.products)

    sample_embeddings = embedding_service.generate_embeddings(
        [faiss_service._product_text(product) for product in items[:1000]]
    )
    checkpoint("after_embedding_1000", embedding_rows=len(sample_embeddings))
    del sample_embeddings

    _, rebuild_peak = run_with_rss_peak(lambda: faiss_service.replace_products(items))
    checkpoint(
        "after_faiss_rebuild",
        indexed=faiss_service.index.ntotal,
        metadata_entries=len(faiss_service.products_by_id),
        rebuild_peak_rss_mb=round(rebuild_peak, 2),
    )

    query = embedding_service.generate_embedding("Produit 1 categorie-1")
    faiss_service.search_similar(query, 5)
    checkpoint("after_one_search")

    for _ in range(100):
        faiss_service.search_similar(query, 5)
    checkpoint("after_100_searches")

    faiss_service.add_product({**items[0], "description": "revision 0"})
    checkpoint("after_one_webhook_update", indexed=faiss_service.index.ntotal)
    for revision in range(1, 100):
        faiss_service.add_product({**items[0], "description": f"revision {revision}"})
    checkpoint("after_100_webhook_updates", indexed=faiss_service.index.ntotal)

    original_get_time = backup_sync_service.get_last_sync_time
    original_fetch = backup_sync_service.fetch_products_updated_after
    original_set_time = backup_sync_service.set_last_sync_time
    backup_sync_service.get_last_sync_time = lambda: "2026-01-01T00:00:00Z"
    backup_sync_service.fetch_products_updated_after = lambda _: items[:100]
    backup_sync_service.set_last_sync_time = lambda _: None
    try:
        backup_sync_service.run_backup_sync()
        checkpoint("after_one_backup_sync")
        for _ in range(max(0, args.repetitions - 1)):
            backup_sync_service.run_backup_sync()
    finally:
        backup_sync_service.get_last_sync_time = original_get_time
        backup_sync_service.fetch_products_updated_after = original_fetch
        backup_sync_service.set_last_sync_time = original_set_time
    checkpoint("after_repeated_backup_syncs", repetitions=args.repetitions)

    for _ in range(args.repetitions):
        faiss_service.replace_products(items)
    checkpoint(
        "after_repeated_full_rebuilds",
        repetitions=args.repetitions,
        indexed=faiss_service.index.ntotal,
    )


if __name__ == "__main__":
    main()
