import json
import os
from datetime import datetime, timezone

STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "sync_state.json")


def get_last_sync_time() -> str | None:
    """Retourne le timestamp ISO de la derniere synchronisation reussie.
    Si aucune synchro n'a jamais eu lieu, retourne None."""
    if not os.path.exists(STATE_FILE):
        return None

    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)
    return state.get("last_sync_time")


def set_last_sync_time(timestamp_iso: str = None):
    """Sauvegarde le timestamp de synchronisation reussie sur disque."""
    timestamp_iso = timestamp_iso or datetime.now(timezone.utc).isoformat()
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_sync_time": timestamp_iso}, f)
