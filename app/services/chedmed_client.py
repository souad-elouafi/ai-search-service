import psycopg2
from psycopg2.extras import RealDictCursor

DB_CONFIG = {
    "dbname": "chedmed_test_db",
    "user": "postgres",
    "password": "souad123",
    "host": "127.0.0.1",
    "port": "5432"
}

def get_all_products(page: int = 1, limit: int = 500):
    """Récupère tous les produits directement depuis PostgreSQL."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("SELECT * FROM products;")
    products = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return {"items": products, "total": len(products)}

def get_product_by_id(product_id: str):
    """Récupère un produit spécifique par son ID."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("SELECT * FROM products WHERE id = %s;", (product_id,))
    product = cursor.fetchone()
    
    cursor.close()
    conn.close()
    return product

def fetch_products_updated_after(since_timestamp=None):
    """Récupère les produits mis à jour après une date/timestamp donnée."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    if since_timestamp:
        cursor.execute("SELECT * FROM products WHERE updated_at > %s;", (since_timestamp,))
    else:
        cursor.execute("SELECT * FROM products;")
        
    products = cursor.fetchall()
    
    cursor.close()
    conn.close()
    return products

# Alias de compatibilité pour les autres services (webhook, faiss, backup_sync)
fetch_product = get_product_by_id
fetch_all_products = get_all_products