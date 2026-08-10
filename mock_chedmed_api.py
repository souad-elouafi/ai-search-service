from dotenv import load_dotenv
load_dotenv()

import os
from datetime import datetime
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(title="Mock ChedMed API (test local uniquement)")

DB_CONFIG = dict(
    host="localhost",
    port=5432,
    dbname="chedmed_test",
    user="postgres",
    password=os.getenv("TEST_DB_PASSWORD", ""),
)


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def row_to_product(row) -> dict:
    """Convertit une ligne de la table products au format attendu par le service IA."""
    return {
        "id": str(row["id"]),
        "title": row.get("title_en") or row.get("title_fr") or row.get("title_ar") or "",
        "description": row.get("description_en") or row.get("description_fr") or row.get("description_ar") or "",
        "category": str(row["category_id"]) if row.get("category_id") is not None else None,
        "brand": str(row["brand_id"]) if row.get("brand_id") is not None else None,
        "color": str(row["color_id"]) if row.get("color_id") is not None else None,
        "condition": str(row["condition_id"]) if row.get("condition_id") is not None else None,
        "price": float(row["price"]) if row.get("price") is not None else None,
        "currency": "MAD",
        "imageUrls": [row["thumbnail_url"]] if row.get("thumbnail_url") else [],
        "status": row.get("status"),
        "isSold": bool(row.get("is_sold")),
        "updatedAt": row["updated_at"].isoformat() if row.get("updated_at") else None,
    }


@app.get("/internal/ai/products")
def list_products(page: int = 1, limit: int = 500, updatedAfter: str = Query(None)):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    offset = (page - 1) * limit

    if updatedAfter:
        cur.execute(
            "SELECT * FROM products WHERE updated_at > %s ORDER BY id LIMIT %s OFFSET %s",
            (updatedAfter, limit, offset),
        )
    else:
        cur.execute(
            "SELECT * FROM products ORDER BY id LIMIT %s OFFSET %s",
            (limit, offset),
        )

    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {"products": [row_to_product(r) for r in rows]}


@app.get("/internal/ai/products/{product_id}")
def get_product(product_id: str):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM products WHERE id = %s", (product_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Product not found")
    return row_to_product(row)