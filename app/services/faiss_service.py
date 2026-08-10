import faiss
import numpy as np
import json
import os
import threading
from app.services.embedding_service import generate_embedding

DIMENSION = 384  # dimension du modele paraphrase-multilingual-MiniLM-L12-v2

index = None
products_by_id = {}
id_map = {}
reverse_id_map = {}
next_internal_id = 1
lock = threading.Lock()


def _product_text(product: dict) -> str:
    name = product.get("name") or product.get("title", "")
    description = product.get("description", "")
    category = product.get("category", "")
    return f"{name} {description} {category}"


def _init_empty_index():
    global index
    base_index = faiss.IndexFlatIP(DIMENSION)
    index = faiss.IndexIDMap2(base_index)


def _reset_state():
    global products_by_id, id_map, reverse_id_map, next_internal_id
    _init_empty_index()
    products_by_id = {}
    id_map = {}
    reverse_id_map = {}
    next_internal_id = 1


def build_index():
    """Construit l'index a partir du catalogue LOCAL (products.json).
    Utilise uniquement en fallback si l'API ChedMed n'est pas encore configuree."""
    _reset_state()

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "products.json")
    with open(data_path, "r", encoding="utf-8") as f:
        raw_products = json.load(f)

    for product in raw_products:
        add_product(product, _rebuild_mode=True)

    print(f"Index FAISS construit (source locale) avec {index.ntotal} produits.")


def sync_from_chedmed_api():
    """Synchronisation initiale complete depuis l'API ChedMed (paginee).
    Remplace entierement l'index actuel."""
    from app.services.chedmed_client import fetch_all_products

    _reset_state()
    page = 1
    total_added = 0

    while True:
        data = fetch_all_products(page=page, limit=500)

        if isinstance(data, list):
            products = data
        else:
            products = data.get("products") or data.get("items") or data.get("data") or []

        if not products:
            break

        for product in products:
            add_product(product, _rebuild_mode=True)
            total_added += 1

        if len(products) < 500:
            break
        page += 1

    print(f"Synchronisation initiale ChedMed terminee : {total_added} produits, {page} page(s).")
    return {"status": "synced", "total_products": total_added, "pages": page}


def add_product(product: dict, _rebuild_mode: bool = False) -> dict:
    global next_internal_id
    product_id = str(product["id"])

    with lock:
        already_exists = product_id in id_map

    if already_exists:
        return update_product(product)

    text = _product_text(product)
    embedding = generate_embedding(text)
    vector = np.array([embedding]).astype("float32")

    with lock:
        internal_id = next_internal_id
        next_internal_id += 1

        index.add_with_ids(vector, np.array([internal_id]).astype("int64"))
        id_map[product_id] = internal_id
        reverse_id_map[internal_id] = product_id
        products_by_id[product_id] = product

    if not _rebuild_mode:
        print(f"Produit {product_id} ajoute a l'index (id interne {internal_id}).")
    return {"status": "added", "product_id": product_id}


def update_product(product: dict) -> dict:
    product_id = str(product["id"])

    with lock:
        old_internal_id = id_map.get(product_id)
        if old_internal_id is not None:
            index.remove_ids(np.array([old_internal_id]).astype("int64"))
            del id_map[product_id]
            del reverse_id_map[old_internal_id]
            del products_by_id[product_id]

    add_product(product)
    print(f"Produit {product_id} mis a jour dans l'index.")
    return {"status": "updated", "product_id": product_id}


def remove_product(product_id) -> dict:
    product_id = str(product_id)

    with lock:
        internal_id = id_map.get(product_id)
        if internal_id is None:
            return {"status": "not_found", "product_id": product_id}

        index.remove_ids(np.array([internal_id]).astype("int64"))
        del id_map[product_id]
        del reverse_id_map[internal_id]
        del products_by_id[product_id]

    print(f"Produit {product_id} retire de l'index.")
    return {"status": "removed", "product_id": product_id}


def search_similar(query_embedding, top_k: int = 5) -> list:
    query_array = np.array([query_embedding]).astype("float32")
    scores, internal_ids = index.search(query_array, top_k)

    results = []
    for score, internal_id in zip(scores[0], internal_ids[0]):
        if internal_id == -1:
            continue
        product_id = reverse_id_map.get(int(internal_id))
        if product_id is None:
            continue
        product = products_by_id[product_id]
        results.append({**product, "similarity_score": float(score)})
    return results


def get_all_products() -> list:
    return list(products_by_id.values())