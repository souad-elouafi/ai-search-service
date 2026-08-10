import requests

from app.config import (
    CHEDMED_API_BASE_URL,
    CHEDMED_API_KEY,
)

def get_product_by_id(product_id: str):

    url = f"{CHEDMED_API_BASE_URL}/internal/ai/products/{product_id}"

    headers = {}

    if CHEDMED_API_KEY:
        headers["x-api-key"] = CHEDMED_API_KEY

    response = requests.get(
        url,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    return response.json() 