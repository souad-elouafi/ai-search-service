from datetime import datetime, timezone
from app.services import faiss_service
from app.services.chedmed_client import fetch_products_updated_after
from app.services.sync_state import get_last_sync_time, set_last_sync_time


def run_backup_sync() -> dict:
    """Recupere tous les produits modifies depuis la derniere synchro reussie,
    et met a jour l'index en consequence. A appeler periodiquement (ex: toutes les 15-30 min)."""

    last_sync = get_last_sync_time()
    data = fetch_products_updated_after(last_sync)

    if isinstance(data, list):
        products = data
    else:
        products = data.get("products") or data.get("items") or data.get("data") or []

    updated_count = 0
    for product in products:
        is_sold = product.get("isSold", False)
        status = product.get("status", "ACTIVE")

        if is_sold or status in ("SOLD", "DEACTIVATED", "DELETED"):
            faiss_service.remove_product(product["id"])
        else:
            faiss_service.add_product(product)  # add_product gere aussi la mise a jour si deja existant

        updated_count += 1

    new_sync_time = datetime.now(timezone.utc).isoformat()
    set_last_sync_time(new_sync_time)

    print(f"Synchronisation de secours terminee : {updated_count} produits traites depuis {last_sync}.")
    return {
        "status": "backup_sync_complete",
        "products_processed": updated_count,
        "synced_since": last_sync,
        "new_sync_time": new_sync_time,
    }