import gc
import weakref
from pathlib import Path

import numpy as np

from app.services import embedding_service, faiss_service


ROOT = Path(__file__).resolve().parents[1]


def make_products(count):
    return [
        {"id": str(i), "name": f"product {i}", "description": "description", "category": "test"}
        for i in range(count)
    ]


def fake_vectors(texts):
    vectors = np.zeros((len(texts), faiss_service.DIMENSION), dtype="float32")
    vectors[:, 0] = 1.0
    return vectors


def test_embedding_uses_inference_mode(monkeypatch):
    inference_states = []

    class Model:
        def encode(self, texts, **kwargs):
            import torch
            inference_states.append(torch.is_inference_mode_enabled())
            return fake_vectors(texts)

    monkeypatch.setattr(embedding_service, "_model", Model())
    embedding_service.generate_embeddings(["one", "two"])
    assert inference_states == [True]


def test_full_rebuild_uses_bounded_embedding_batches(monkeypatch):
    sizes = []

    def recording(texts):
        sizes.append(len(texts))
        return fake_vectors(texts)

    monkeypatch.setattr(faiss_service, "generate_embeddings", recording)
    faiss_service.replace_products(make_products(1_201))
    assert sizes == [500, 500, 201]
    assert max(sizes) <= faiss_service.AI_INDEX_BUILD_BATCH_SIZE


def test_temporary_embedding_arrays_are_released(monkeypatch):
    references = []

    def recording(texts):
        vectors = fake_vectors(texts)
        references.append(weakref.ref(vectors))
        return vectors

    monkeypatch.setattr(faiss_service, "generate_embeddings", recording)
    faiss_service.replace_products(make_products(33))
    gc.collect()
    assert references
    assert all(reference() is None for reference in references)


def test_repeated_rebuild_does_not_retain_application_state(monkeypatch):
    monkeypatch.setattr(faiss_service, "generate_embeddings", fake_vectors)
    for _ in range(5):
        faiss_service.replace_products(make_products(100))
        assert len(faiss_service.products_by_id) == 100
        assert len(faiss_service.id_map) == 100
        assert len(faiss_service.reverse_id_map) == 100
        assert faiss_service.index.ntotal == 100


def test_startup_local_index_is_disabled_by_default(monkeypatch):
    import main

    called = []
    existing_model = object()
    monkeypatch.setattr(embedding_service, "_model", existing_model)
    monkeypatch.setattr(main, "AI_BUILD_LOCAL_INDEX_ON_STARTUP", False)
    monkeypatch.setattr(main, "build_index", lambda: called.append("index"))
    monkeypatch.setattr(main, "start_scheduler", lambda: called.append("scheduler"))
    main.startup_event()
    assert called == ["scheduler"]
    assert embedding_service._model is existing_model


def test_production_command_has_one_worker_and_no_reload():
    command = (ROOT / "Procfile").read_text(encoding="utf-8-sig").strip()
    assert command.startswith("web: uvicorn main:app")
    assert "--reload" not in command
    assert "--workers 1" in command
    assert "--workers 2" not in command
    assert "gunicorn" not in command.lower()


def test_no_repository_restart_manager_configuration():
    deployment_files = list(ROOT.glob("Dockerfile*"))
    deployment_files += list(ROOT.glob("*compose*.yml"))
    deployment_files += list(ROOT.glob("*.service"))
    deployment_files += list(ROOT.glob("ecosystem*.js"))
    assert deployment_files == []
