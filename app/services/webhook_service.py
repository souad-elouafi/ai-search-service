from app.services import faiss_service
from app.services.chedmed_client import fetch_product
import threading

# Anti-doublon : garde en memoire les eventId deja traites.
# Limitation connue : reinitialise a chaque redemarrage du serveur.
# Pour une vraie production, remplacer par un stockage persistant (fichier ou DB).
processed_event_ids = set()
_event_lock = threading.Lock()

REMOVE_FROM_SEARCH_EVENTS = {"PRODUCT_SOLD", "PRODUCT_DEACTIVATED", "PRODUCT_DELETED"}
ADD_OR_UPDATE_EVENTS = {"PRODUCT_CREATED", "PRODUCT_UPDATED", "PRODUCT_REACTIVATED"}


def handle_webhook_event(event_id: str, event_type: str, product_id: str) -> dict:
    """Traite un evenement recu du webhook ChedMed."""

    with _event_lock:
        if event_id in processed_event_ids:
            return {"status": "duplicate_ignored", "event_id": event_id}
        processed_event_ids.add(event_id)

    try:
        if event_type in ADD_OR_UPDATE_EVENTS:
            product = fetch_product(product_id)
            result = faiss_service.add_product(product)
            return {"status": "processed", "event_type": event_type, "action": result}

        if event_type in REMOVE_FROM_SEARCH_EVENTS:
            result = faiss_service.remove_product(product_id)
            return {"status": "processed", "event_type": event_type, "action": result}

        return {"status": "ignored", "reason": f"Type d'evenement inconnu: {event_type}"}
    except Exception:
        # Permit a genuine retry when processing failed before acknowledgement.
        with _event_lock:
            processed_event_ids.discard(event_id)
        raise
