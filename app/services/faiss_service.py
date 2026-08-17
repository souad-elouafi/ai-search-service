import json
import os
import threading

from app.config import AI_INDEX_BUILD_BATCH_SIZE, AI_TORCH_NUM_THREADS

# Import FAISS only after configuration is loaded so operator-provided OpenMP
# settings are visible when its native runtime initializes.
import faiss
import numpy as np

from app.services.embedding_service import generate_embeddings
from app.services.sync_state import set_last_sync_time

DIMENSION = 384
faiss.omp_set_num_threads(AI_TORCH_NUM_THREADS)

index = None
products_by_id = {}
id_map = {}
reverse_id_map = {}
next_internal_id = 1
lock = threading.RLock()
sync_lock = threading.Lock()


def _product_text(product: dict) -> str:
    name = product.get("name") or product.get("title", "")
    description = product.get("description", "")
    category = product.get("category", "")
    return f"{name} {description} {category}"


def _empty_index():
    base_index = faiss.IndexFlatIP(DIMENSION)
    mapped_index = faiss.IndexIDMap2(base_index)
    # SWIG does not transfer ownership automatically. Transfer it explicitly so
    # IndexIDMap2 keeps the wrapped native index alive without double deletion.
    mapped_index.own_fields = True
    base_index.this.disown()
    return mapped_index


def _reset_state():
    global index, products_by_id, id_map, reverse_id_map, next_internal_id
    with lock:
        index = _empty_index()
        products_by_id = {}
        id_map = {}
        reverse_id_map = {}
        next_internal_id = 1


class _StagingState:
    """Incrementally build a complete index while keeping the live one intact."""

    def __init__(self):
        self.index = _empty_index()
        self.products = {}
        self.id_map = {}
        self.reverse_id_map = {}
        self.next_internal_id = 1

    def add_batch(self, products: list[dict]):
        # Last occurrence wins within a batch, matching the previous behavior.
        unique = {str(product["id"]): product for product in products}
        product_ids = list(unique)
        batch_products = [unique[product_id] for product_id in product_ids]
        embeddings = generate_embeddings([_product_text(product) for product in batch_products])
        vectors = np.asarray(embeddings, dtype="float32")
        if batch_products and (vectors.ndim != 2 or vectors.shape[1] != DIMENSION):
            raise ValueError(f"Dimension d'embedding invalide: {vectors.shape}")

        duplicate_ids = [self.id_map[product_id] for product_id in product_ids if product_id in self.id_map]
        if duplicate_ids:
            self.index.remove_ids(np.asarray(duplicate_ids, dtype="int64"))
            for internal_id in duplicate_ids:
                old_product_id = self.reverse_id_map.pop(internal_id)
                self.id_map.pop(old_product_id, None)

        internal_ids = np.arange(
            self.next_internal_id,
            self.next_internal_id + len(product_ids),
            dtype="int64",
        )
        if batch_products:
            self.index.add_with_ids(vectors, internal_ids)
        for product_id, product, internal_id in zip(
            product_ids, batch_products, internal_ids.tolist()
        ):
            self.products[product_id] = product
            self.id_map[product_id] = internal_id
            self.reverse_id_map[internal_id] = product_id
        self.next_internal_id += len(product_ids)

        # Do not retain full-catalogue text or embedding arrays between batches.
        del embeddings, vectors, internal_ids, batch_products, unique

    def result(self):
        return (
            self.index,
            self.products,
            self.id_map,
            self.reverse_id_map,
            self.next_internal_id,
        )


def _iter_batches(products, batch_size: int = AI_INDEX_BUILD_BATCH_SIZE):
    batch = []
    for product in products:
        batch.append(product)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _build_state(products):
    """Build a replacement state with embeddings bounded to one small batch."""
    staging = _StagingState()
    for batch in _iter_batches(products):
        staging.add_batch(batch)
    return staging.result()


def replace_products(products: list[dict]) -> int:
    """Build outside the lock, then atomically expose the completed index."""
    global index, products_by_id, id_map, reverse_id_map, next_internal_id
    state = _build_state(products)
    with lock:
        index, products_by_id, id_map, reverse_id_map, next_internal_id = state
        return index.ntotal


def build_index():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "products.json")
    with open(data_path, "r", encoding="utf-8") as f:
        raw_products = json.load(f)
    count = replace_products(raw_products)
    print(f"Index FAISS construit (source locale) avec {count} produits.")


def sync_from_chedmed_api():
    from app.services.chedmed_client import fetch_all_products

    if not sync_lock.acquire(blocking=False):
        return {"status": "sync_already_running"}
    try:
        page = 1
        staging = _StagingState()
        while True:
            data = fetch_all_products(page=page, limit=500)
            products = data if isinstance(data, list) else (
                data.get("products") or data.get("items") or data.get("data") or []
            )
            if not products:
                break
            for batch in _iter_batches(products):
                staging.add_batch(batch)
            if len(products) < 500:
                break
            page += 1
        global index, products_by_id, id_map, reverse_id_map, next_internal_id
        state = staging.result()
        with lock:
            index, products_by_id, id_map, reverse_id_map, next_internal_id = state
            total = index.ntotal
        set_last_sync_time()
        print(f"Synchronisation initiale ChedMed terminee : {total} produits, {page} page(s).")
        return {"status": "synced", "total_products": total, "pages": page}
    finally:
        sync_lock.release()


def add_products(products: list[dict]) -> dict:
    """Batch add/update products, skipping unchanged searchable text safely."""
    global next_internal_id
    unique = {str(product["id"]): product for product in products}
    with lock:
        changed = [
            product for product_id, product in unique.items()
            if product_id not in products_by_id
            or _product_text(products_by_id[product_id]) != _product_text(product)
        ]
        metadata_only = [
            (product_id, product) for product_id, product in unique.items()
            if product_id in products_by_id
            and _product_text(products_by_id[product_id]) == _product_text(product)
        ]

    embeddings = (
        generate_embeddings([_product_text(product) for product in changed])
        if changed else []
    )
    vectors = np.asarray(embeddings, dtype="float32")
    if changed and (vectors.ndim != 2 or vectors.shape[1] != DIMENSION):
        raise ValueError(f"Dimension d'embedding invalide: {vectors.shape}")

    with lock:
        for product_id, product in metadata_only:
            products_by_id[product_id] = product
        old_ids = [id_map[str(product["id"])] for product in changed if str(product["id"]) in id_map]
        if old_ids:
            index.remove_ids(np.asarray(old_ids, dtype="int64"))
            for internal_id in old_ids:
                product_id = reverse_id_map.pop(internal_id)
                id_map.pop(product_id, None)
        new_ids = np.arange(next_internal_id, next_internal_id + len(changed), dtype="int64")
        if changed:
            index.add_with_ids(vectors, new_ids)
        for product, internal_id in zip(changed, new_ids.tolist()):
            product_id = str(product["id"])
            id_map[product_id] = internal_id
            reverse_id_map[internal_id] = product_id
            products_by_id[product_id] = product
        next_internal_id += len(changed)
    return {"processed": len(unique), "embedded": len(changed), "unchanged": len(metadata_only)}


def add_product(product: dict, _rebuild_mode: bool = False) -> dict:
    product_id = str(product["id"])
    with lock:
        existed = product_id in id_map
        unchanged = existed and _product_text(products_by_id[product_id]) == _product_text(product)
    add_products([product])
    status = "unchanged" if unchanged else ("updated" if existed else "added")
    return {"status": status, "product_id": product_id}


def update_product(product: dict) -> dict:
    return add_product(product)


def remove_product(product_id) -> dict:
    product_id = str(product_id)
    with lock:
        internal_id = id_map.get(product_id)
        if internal_id is None:
            return {"status": "not_found", "product_id": product_id}
        index.remove_ids(np.asarray([internal_id], dtype="int64"))
        del id_map[product_id]
        del reverse_id_map[internal_id]
        del products_by_id[product_id]
    return {"status": "removed", "product_id": product_id}


def search_similar(query_embedding, top_k: int = 5) -> list:
    query_array = np.asarray([query_embedding], dtype="float32")
    with lock:
        scores, internal_ids = index.search(query_array, top_k)
        results = []
        for score, internal_id in zip(scores[0], internal_ids[0]):
            if internal_id == -1:
                continue
            product_id = reverse_id_map.get(int(internal_id))
            if product_id is not None:
                results.append({**products_by_id[product_id], "similarity_score": float(score)})
        return results


def get_all_products() -> list:
    with lock:
        return list(products_by_id.values())


_reset_state()
