from datetime import datetime, timezone
from app.services import faiss_service
from app.services.chedmed_client import fetch_products_updated_after
from app.services.sync_state import get_last_sync_time, set_last_sync_time


def run_backup_sync() -> dict:
    """Recupere tous les produits modifies depuis la derniere synchro reussie,
    et met a jour l'index en consequence. A appeler periodiquement (ex: toutes les 15-30 min)."""

    last_sync = get_last_sync_time()
    if not last_sync:
        return {
            "status": "not_initialized",
            "products_processed": 0,
            "message": "Run the explicit initial catalogue sync before backup sync.",
        }

    if not faiss_service.sync_lock.acquire(blocking=False):
        return {"status": "sync_already_running", "products_processed": 0}
    sync_started_at = datetime.now(timezone.utc).isoformat()
    try:
        data = fetch_products_updated_after(last_sync)

        if isinstance(data, list):
            products = data
        else:
            products = data.get("products") or data.get("items") or data.get("data") or []

        active_products = []
        removed_count = 0
        for product in products:
            is_sold = product.get("isSold", False)
            status = product.get("status", "ACTIVE")

            if is_sold or status in ("SOLD", "DEACTIVATED", "DELETED"):
                faiss_service.remove_product(product["id"])
                removed_count += 1
            else:
                active_products.append(product)

        batch_result = faiss_service.add_products(active_products)
        updated_count = len(products)

        set_last_sync_time(sync_started_at)

        print(f"Synchronisation de secours terminee : {updated_count} produits traites depuis {last_sync}.")
        return {
            "status": "backup_sync_complete",
            "products_processed": updated_count,
            "products_embedded": batch_result["embedded"],
            "products_removed": removed_count,
            "synced_since": last_sync,
            "new_sync_time": sync_started_at,
        }
    finally:
        faiss_service.sync_lock.release()
