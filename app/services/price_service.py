import numpy as np
from app.services.embedding_service import generate_embedding
from app.services import faiss_service


def estimate_price(description: str, top_k: int = 10) -> dict:
    """Estime un prix suggéré en comparant à des produits similaires déjà indexés."""
    results = faiss_service.search_similar(generate_embedding(description), top_k=top_k)

    if not results:
        return {
            "suggested_price": None,
            "mean_price": None,
            "price_range": None,
            "comparable_products": [],
            "message": "Aucun produit similaire trouvé pour estimer un prix.",
        }

    prices = [r["price"] for r in results if r.get("price") is not None and r["price"] > 0]
    
    if not prices:
        return {
            "suggested_price": None,
            "mean_price": None,
            "price_range": None,
            "comparable_products": [],
            "message": "Les produits similaires trouvés ne possèdent pas de prix valide.",
        }

    comparable_products = [
        {
            "name": r.get("name") or r.get("title") or "Produit",
            "price": r["price"],
            "similarity_score": r.get("similarity_score", 0),
        }
        for r in results if r.get("price") is not None and r["price"] > 0
    ]

    prices_array = np.array(prices)
    suggested_price = float(np.median(prices_array))
    mean_price = float(np.mean(prices_array))
    price_min = float(np.min(prices_array))
    price_max = float(np.max(prices_array))

    return {
        "suggested_price": round(suggested_price, 2),
        "mean_price": round(mean_price, 2),
        "price_range": {"min": round(price_min, 2), "max": round(price_max, 2)},
        "comparable_products": comparable_products,
        "based_on_n_products": len(prices),
    }


def check_price_alert(description: str, category: str, seller_price: float, top_k: int = 30) -> dict:
    """
    Vérifie si le prix saisi par le vendeur est cohérent avec le marché,
    en comparant aux produits similaires de la même catégorie (avec fallback dynamique).
    """
    results = faiss_service.search_similar(generate_embedding(f"{category} {description}"), top_k=top_k)

    target_category = category.strip().lower()

    # 1. Filtre principal : correspondance de catégorie (exacte ou partielle)
    same_category_prices = [
        r["price"] for r in results
        if r.get("price") is not None and r["price"] > 0 and (
            target_category in r.get("category", "").strip().lower() or
            r.get("category", "").strip().lower() in target_category
        )
    ]

    # 2. Premier fallback : si moins de 3 produits trouvés dans FAISS, chercher dans tout le catalogue
    if len(same_category_prices) < 3:
        all_products = faiss_service.get_all_products()
        same_category_prices = [
            p["price"] for p in all_products
            if p.get("price") is not None and p["price"] > 0 and (
                target_category in p.get("category", "").strip().lower() or
                p.get("category", "").strip().lower() in target_category
            )
        ]

    # 3. Second fallback : si la catégorie n'existe pas dans le catalogue de test, prendre tous les voisins FAISS proches
    if len(same_category_prices) < 3:
        same_category_prices = [
            r["price"] for r in results
            if r.get("price") is not None and r["price"] > 0
        ]

    # Si la base est totalement vide
    if not same_category_prices:
        return {
            "alert": "no_data",
            "message": "Aucun produit trouvé dans la base pour comparer les prix.",
            "seller_price": seller_price,
            "market_stats": None,
            "based_on_n_products": 0,
        }

    prices_array = np.array(same_category_prices)
    median_price = float(np.median(prices_array))
    mean_price = float(np.mean(prices_array))
    p25 = float(np.percentile(prices_array, 25))
    p75 = float(np.percentile(prices_array, 75))
    min_price = float(np.min(prices_array))
    max_price = float(np.max(prices_array))

    low_threshold = median_price * 0.7
    high_threshold = median_price * 1.5

    if seller_price < low_threshold:
        alert = "too_low"
        message = (
            f"Prix bas : Le prix proposé ({seller_price} MAD) est inférieur au marché pour la catégorie '{category}' "
            f"(Prix moyen : {round(mean_price, 2)} MAD, Médiane : {round(median_price, 2)} MAD)."
        )
    elif seller_price > high_threshold:
        alert = "too_high"
        message = (
            f"Prix élevé : Le prix proposé ({seller_price} MAD) est supérieur au marché pour la catégorie '{category}' "
            f"(Prix moyen : {round(mean_price, 2)} MAD, Médiane : {round(median_price, 2)} MAD)."
        )
    else:
        alert = "normal"
        message = (
            f"Prix cohérent : Le prix proposé ({seller_price} MAD) est aligné avec la moyenne pour la catégorie '{category}' "
            f"(Prix moyen : {round(mean_price, 2)} MAD)."
        )

    return {
        "alert": alert,
        "message": message,
        "seller_price": seller_price,
        "market_stats": {
            "mean": round(mean_price, 2),
            "median": round(median_price, 2),
            "p25": round(p25, 2),
            "p75": round(p75, 75) if False else round(p75, 2),
            "min": round(min_price, 2),
            "max": round(max_price, 2),
        },
        "based_on_n_products": len(same_category_prices),
    }