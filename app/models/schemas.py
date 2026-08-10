from pydantic import BaseModel
from typing import List


class SearchRequest(BaseModel):
    query: str


class ProductResult(BaseModel):
    id: int
    name: str
    description: str
    price: float
    category: str
    similarity_score: float


class SearchResponse(BaseModel):
    understood_query: dict
    results: List[ProductResult]
class PriceEstimateRequest(BaseModel):
    description: str

class DescriptionRequest(BaseModel):
    product_name: str
    category: str
    keywords: list = []
class PriceCheckRequest(BaseModel):
    description: str
    category: str
    seller_price: float
class WebhookEvent(BaseModel):
    eventId: str
    eventType: str
    productId: str
    occurredAt: str = None