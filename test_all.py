import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def log_test(name, success, details=""):
    status = "SUCCESS" if success else "FAILED"
    print(f"[{status}] {name}")
    if details:
        print(f"   -> {details}")

print("=== Lancement de la vérification globale du microservice IA ===")

# 1. Test Healthcheck
try:
    r = requests.get(f"{BASE_URL}/health")
    log_test("1. Health Check", r.status_code == 200, r.json())
except Exception as e:
    log_test("1. Health Check", False, f"Impossible de contacter le serveur : {e}")
    exit(1)

# 2. Test Synchronisation Catalogue Initial
r = requests.post(f"{BASE_URL}/admin/sync-catalogue")
log_test("2. Synchronisation Catalogue", r.status_code == 200, r.json())

# 3. Test Recherche Texte
payload_search = {"query": "t-shirt blanc en coton"}
r = requests.post(f"{BASE_URL}/api/search/text", json=payload_search)
log_test("3. Recherche Texte", r.status_code == 200, f"Résultats trouvés : {len(r.json().get('results', []))}")

# 4. Test Estimation de prix
payload_price = {"description": "t-shirt blanc coton de marque"}
r = requests.post(f"{BASE_URL}/api/seller/estimate-price", json=payload_price)
log_test("4. Estimation Prix Vendeur", r.status_code == 200, f"Prix suggéré : {r.json().get('suggested_price')} MAD")

# 5. Test Alerte de prix
payload_alert = {
    "description": "t-shirt blanc",
    "category": "vetement",
    "seller_price": 500
}
r = requests.post(f"{BASE_URL}/api/seller/check-price", json=payload_alert)
log_test("5. Alerte Prix Vendeur", r.status_code == 200, f"Statut : {r.json().get('alert')} | Message : {r.json().get('message')}")

# 6. Test Webhook - Anti-doublon (eventId)
webhook_event = {
    "eventId": "evt_test_999",
    "eventType": "PRODUCT_SOLD",
    "productId": "prod_123"
}
r1 = requests.post(f"{BASE_URL}/catalogue/webhook", json=webhook_event)
r2 = requests.post(f"{BASE_URL}/catalogue/webhook", json=webhook_event)  # Envoi du doublon

is_duplicate_handled = r2.status_code == 200 and r2.json().get("status") == "duplicate_ignored"
log_test("6. Webhook & Anti-doublon (eventId)", is_duplicate_handled, f"Second envoi : {r2.json()}")

print("\n=== Vérification terminée ! ===")