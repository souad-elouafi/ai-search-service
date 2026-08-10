import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from app.models.schemas import (
    SearchRequest,
    PriceEstimateRequest,
    DescriptionRequest,
    PriceCheckRequest,
    WebhookEvent,
)
from app.services.text_service import understand_query, understand_audio_query
from app.services.embedding_service import generate_embedding
from app.services import faiss_service
from app.services.faiss_service import build_index, search_similar
from app.services.voice_service import transcribe_dual
from app.services.image_service import describe_image
from app.services.price_service import estimate_price, check_price_alert
from app.services.description_service import suggest_description
from app.services.webhook_service import handle_webhook_event
from app.config import CHEDMED_API_BASE_URL
from app.services.backup_sync_service import run_backup_sync
from app.services.scheduler import start_scheduler, stop_scheduler

app = FastAPI(title="AI Search Service - Chedmed")


def run_search_pipeline(text_for_understanding: str):
    understood = understand_query(text_for_understanding)
    query_embedding = generate_embedding(understood["search_text"])
    results = search_similar(query_embedding, top_k=5)
    return understood, results


@app.post("/api/search/text")
def search_text(request: SearchRequest):
    try:
        understood, results = run_search_pipeline(request.query)
        return {"understood_query": understood, "results": results}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service IA temporairement indisponible: {str(e)}")


@app.post("/api/search/audio")
async def search_audio(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        transcriptions = transcribe_dual(temp_path)
        understood = understand_audio_query(
            transcriptions["whisper_text"],
            transcriptions["gemini_text"],
        )
        query_embedding = generate_embedding(understood["search_text"])
        results = search_similar(query_embedding, top_k=5)

        return {
            "whisper_text": transcriptions["whisper_text"],
            "gemini_text": transcriptions["gemini_text"],
            "understood_query": understood,
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service IA temporairement indisponible: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/search/image")
async def search_image(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        description = describe_image(temp_path)
        understood, results = run_search_pipeline(description)

        return {
            "image_description": description,
            "understood_query": understood,
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service IA temporairement indisponible: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/seller/estimate-price")
def seller_estimate_price(request: PriceEstimateRequest):
    try:
        result = estimate_price(request.description)
        return result
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service IA temporairement indisponible: {str(e)}")


@app.post("/api/seller/check-price")
def seller_check_price(request: PriceCheckRequest):
    try:
        result = check_price_alert(request.description, request.category, request.seller_price)
        return result
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service IA temporairement indisponible: {str(e)}")


@app.post("/api/seller/suggest-description")
async def seller_suggest_description(
    product_name: str = Form(...),
    category: str = Form(...),
    keywords: str = Form(""),
    language: str = Form("fr"),
    image: UploadFile = File(None),
):
    temp_path = None
    try:
        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]

        if image is not None:
            temp_path = f"temp_{image.filename}"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

        descriptions = suggest_description(
            product_name, category, keywords_list, language, temp_path
        )
        return descriptions
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service IA temporairement indisponible: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/catalogue/webhook")
def catalogue_webhook(event: WebhookEvent):
    try:
        result = handle_webhook_event(event.eventId, event.eventType, event.productId)
        return {"acknowledged": True, **result}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Erreur traitement webhook: {str(e)}")


@app.post("/admin/sync-catalogue")
def admin_sync_catalogue():
    try:
        if not CHEDMED_API_BASE_URL:
            raise HTTPException(status_code=400, detail="CHEDMED_API_BASE_URL non configure dans .env")
        result = faiss_service.sync_from_chedmed_api()
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Erreur synchronisation: {str(e)}")


@app.post("/admin/backup-sync")
def admin_backup_sync():
    try:
        if not CHEDMED_API_BASE_URL:
            raise HTTPException(status_code=400, detail="CHEDMED_API_BASE_URL non configure dans .env")
        result = run_backup_sync()
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Erreur synchronisation de secours: {str(e)}")


@app.on_event("startup")
def startup_event():
    build_index()
    start_scheduler()


@app.on_event("shutdown")
def shutdown_event():
    stop_scheduler()


@app.get("/health")
def health_check():
    return {"status": "ok"}