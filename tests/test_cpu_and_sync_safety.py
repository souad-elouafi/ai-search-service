import hashlib
import hmac
import json
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.services import embedding_service, faiss_service


def fake_embeddings(texts):
    vectors = np.zeros((len(texts), faiss_service.DIMENSION), dtype="float32")
    for row, text in enumerate(texts):
        vectors[row, int(hashlib.sha256(text.encode()).hexdigest(), 16) % faiss_service.DIMENSION] = 1.0
    return vectors


@pytest.fixture(autouse=True)
def reset_index(monkeypatch):
    monkeypatch.setattr(faiss_service, "generate_embeddings", fake_embeddings)
    faiss_service._reset_state()


def products():
    return [
        {"id": "a", "name": "red shoe", "description": "sport", "category": "shoes"},
        {"id": "b", "name": "blue shirt", "description": "cotton", "category": "clothes"},
        {"id": "c", "name": "green bag", "description": "leather", "category": "bags"},
    ]


def test_model_is_initialized_once(monkeypatch):
    created = []

    class FakeModel:
        def __init__(self, name):
            created.append(name)

    monkeypatch.setattr(embedding_service, "SentenceTransformer", FakeModel)
    monkeypatch.setattr(embedding_service, "_model", None)
    assert embedding_service.get_model() is embedding_service.get_model()
    assert created == [embedding_service.MODEL_NAME]


def test_add_update_remove_search_and_mapping_integrity():
    faiss_service.replace_products(products())
    assert faiss_service.index.ntotal == 3
    assert set(faiss_service.id_map) == {"a", "b", "c"}
    assert all(faiss_service.reverse_id_map[v] == k for k, v in faiss_service.id_map.items())

    assert faiss_service.update_product({**products()[0], "name": "orange shoe"})["status"] == "updated"
    assert faiss_service.remove_product("b")["status"] == "removed"
    assert faiss_service.add_product({"id": "d", "name": "watch"})["status"] == "added"
    assert faiss_service.index.ntotal == 3
    assert set(faiss_service.id_map) == {"a", "c", "d"}


def test_unchanged_search_text_is_not_reembedded(monkeypatch):
    calls = []

    def recording(texts):
        calls.append(list(texts))
        return fake_embeddings(texts)

    monkeypatch.setattr(faiss_service, "generate_embeddings", recording)
    faiss_service.replace_products(products())
    result = faiss_service.add_products([{**products()[0], "price": 99}])
    assert result == {"processed": 1, "embedded": 0, "unchanged": 1}
    assert len(calls) == 1
    assert faiss_service.products_by_id["a"]["price"] == 99


def test_full_sync_batches_and_remains_searchable(monkeypatch):
    calls = []

    def recording(texts):
        calls.append(len(texts))
        return fake_embeddings(texts)

    monkeypatch.setattr(faiss_service, "generate_embeddings", recording)
    monkeypatch.setattr("app.services.chedmed_client.fetch_all_products", lambda page, limit: products() if page == 1 else [])
    monkeypatch.setattr(faiss_service, "set_last_sync_time", lambda: None)
    result = faiss_service.sync_from_chedmed_api()
    assert result["total_products"] == 3
    assert calls == [3]


def test_search_after_batch_rebuild_in_clean_process():
    # On macOS, loading pytest's scientific plugins alongside both PyTorch and
    # FAISS can load conflicting OpenMP runtimes. Validate native search in the
    # same clean-process shape used by production instead.
    code = r'''
import hashlib
import numpy as np
from app.services import faiss_service as service
def embed(texts):
    vectors = np.zeros((len(texts), service.DIMENSION), dtype="float32")
    for row, text in enumerate(texts):
        vectors[row, int(hashlib.sha256(text.encode()).hexdigest(), 16) % service.DIMENSION] = 1.0
    return vectors
service.generate_embeddings = embed
items = [
    {"id": "a", "name": "red shoe", "description": "sport", "category": "shoes"},
    {"id": "b", "name": "blue shirt", "description": "cotton", "category": "clothes"},
]
service.replace_products(items)
query = embed([service._product_text(items[0])])[0]
assert service.search_similar(query, 1)[0]["id"] == "a"
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_backup_sync_without_state_never_fetches_history(monkeypatch):
    from app.services import backup_sync_service

    monkeypatch.setattr(backup_sync_service, "get_last_sync_time", lambda: None)
    monkeypatch.setattr(
        backup_sync_service,
        "fetch_products_updated_after",
        lambda value: pytest.fail(f"unexpected history fetch: {value}"),
    )
    assert backup_sync_service.run_backup_sync()["status"] == "not_initialized"


def test_backup_sync_batches_active_products(monkeypatch):
    from app.services import backup_sync_service

    seen = []
    monkeypatch.setattr(backup_sync_service, "get_last_sync_time", lambda: "2026-01-01T00:00:00Z")
    monkeypatch.setattr(backup_sync_service, "set_last_sync_time", seen.append)
    monkeypatch.setattr(backup_sync_service, "fetch_products_updated_after", lambda _: products())
    result = backup_sync_service.run_backup_sync()
    assert result["products_embedded"] == 3
    assert len(seen) == 1
    assert faiss_service.index.ntotal == 3


def test_scheduler_disabled_and_idempotent(monkeypatch):
    from app.services import scheduler as module

    class FakeScheduler:
        running = False
        jobs = []

        def add_job(self, *args, **kwargs):
            self.jobs.append(kwargs)

        def start(self):
            self.running = True

    fake = FakeScheduler()
    monkeypatch.setattr(module, "scheduler", fake)
    monkeypatch.setattr(module, "ENABLE_BACKUP_SCHEDULER", False)
    assert module.start_scheduler() is False
    assert fake.jobs == []
    monkeypatch.setattr(module, "ENABLE_BACKUP_SCHEDULER", True)
    assert module.start_scheduler() is True
    assert module.start_scheduler() is False
    assert len(fake.jobs) == 1
    assert fake.jobs[0]["max_instances"] == 1


def test_chedmed_client_uses_x_api_key(monkeypatch):
    from app.services import chedmed_client

    captured = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"id": "a"}

    def fake_get(url, **kwargs):
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr(chedmed_client, "CHEDMED_API_KEY", "test-key")
    monkeypatch.setattr(chedmed_client.requests, "get", fake_get)
    chedmed_client.fetch_product("a")
    assert captured["headers"] == {"X-API-Key": "test-key"}
    assert "Authorization" not in captured["headers"]


def test_webhook_hmac_timestamp_and_duplicate(monkeypatch):
    import main
    from app.services import webhook_service

    secret = "unit-test-secret"
    monkeypatch.setattr(main, "AI_WEBHOOK_SECRET", secret)
    monkeypatch.setattr(main, "build_index", lambda: None)
    monkeypatch.setattr(main, "start_scheduler", lambda: False)
    monkeypatch.setattr(main, "stop_scheduler", lambda: None)
    monkeypatch.setattr(webhook_service, "processed_event_ids", set())
    monkeypatch.setattr(webhook_service.faiss_service, "remove_product", lambda product_id: {"status": "removed"})
    # main imported the callable directly, so route it through the patched service.
    monkeypatch.setattr(main, "handle_webhook_event", webhook_service.handle_webhook_event)
    body = json.dumps(
        {"eventId": "evt-1", "eventType": "PRODUCT_DELETED", "productId": "a"},
        separators=(",", ":"),
    ).encode()
    timestamp = str(int(datetime.now(timezone.utc).timestamp()))
    signature = hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "X-ChedMed-Event-Id": "evt-1",
        "X-ChedMed-Timestamp": timestamp,
        "X-ChedMed-Signature": signature,
    }
    with TestClient(main.app) as client:
        first = client.post("/catalogue/webhook", content=body, headers=headers)
        second = client.post("/catalogue/webhook", content=body, headers=headers)
        assert first.status_code == 200
        assert second.json()["status"] == "duplicate_ignored"
        stale_headers = {**headers, "X-ChedMed-Timestamp": "946684800"}
        assert client.post("/catalogue/webhook", content=body, headers=stale_headers).status_code == 401
