import requests
from app.config import CHEDMED_API_BASE_URL, CHEDMED_API_KEY


def fetch_product(product_id: str) -> dict:
    """Recupere les donnees a jour d'UN produit depuis l'API ChedMed."""
    url = f"{CHEDMED_API_BASE_URL}/internal/ai/products/{product_id}"
    headers = {"Authorization": f"Bearer {CHEDMED_API_KEY}"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    return response.json()


def fetch_all_products(page: int = 1, limit: int = 500) -> dict:
    """Recupere une page du catalogue complet (synchronisation initiale)."""
    url = f"{CHEDMED_API_BASE_URL}/internal/ai/products"
    headers = {"Authorization": f"Bearer {CHEDMED_API_KEY}"}
    params = {"page": page, "limit": limit}
    response = requests.get(url, headers=headers, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def fetch_products_updated_after(timestamp_iso: str) -> dict:
    """Recupere les produits modifies depuis un timestamp donne (synchronisation de secours)."""
    url = f"{CHEDMED_API_BASE_URL}/internal/ai/products"
    headers = {"Authorization": f"Bearer {CHEDMED_API_KEY}"}
    params = {"updatedAfter": timestamp_iso}
    response = requests.get(url, headers=headers, params=params, timeout=15)
    response.raise_for_status()
    return response.json()