# -*- coding: utf-8 -*-
"""
FastAPI Persistent Inference Service - Main Application
Integrates MongoDB, Input Validation, Web Retrieval, and Model Prediction.
"""
import datetime
import os
import atexit
import sys
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import uvicorn

# Import refactored modules
from mongodb_manager import MongoDBConfig, MongoDBManager
from input_validator import is_safe_input, validator as input_validator
from model_wrapper import ModelWrapper, ModelConfig

# ===== Logging Configuration =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== Global Instances =====
# These will be initialized in the startup event
model_wrapper: Optional[ModelWrapper] = None
mongodb_manager: Optional[MongoDBManager] = None

# ===== Data Models (Pydantic) =====
class PredictWithContextRequest(BaseModel):
    text: str
    use_retrieval: bool = True
    sources: Optional[List[str]] = None
    save_to_db: bool = Field(default=True, description="Whether to save to MongoDB")

class BatchPredictRequest(BaseModel):
    texts: List[str]

class PredictResponse(BaseModel):
    text: str
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]
    context: Optional[Dict[str, Any]] = None
    retrieval_status: str = Field(..., description="Status of the retrieval process: 'success', 'empty', or 'failed'")
    saved_to_db: Optional[bool] = None
    db_id: Optional[str] = None

class BatchPredictResponse(BaseModel):
    count: int
    results: List[Dict[str, Any]]

class BulkImportRequest(BaseModel):
    data: List[Dict[str, str]] = Field(..., description="List of data, each item must contain 'label' and 'content'")

class BulkImportResponse(BaseModel):
    success: bool
    imported_count: int
    message: str

# ===== FastAPI Application =====
app = FastAPI(
    title="MisRoBERTa Inference Service (RAG Enhanced)",
    description="Persistent FastAPI service for MisRoBERTa model inference, featuring RAG, input validation, and MongoDB integration.",
    version="1.0.0"
)

# --- Startup and Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    global model_wrapper, mongodb_manager
    
    # 1. Initialize MongoDB Manager
    mongodb_manager = MongoDBManager(MongoDBConfig())
    
    # 2. Load Model and Encoders
    # Model paths are expected to be passed via environment variables or command line args
    # For a persistent service, we'll use environment variables or hardcoded defaults for demonstration
    MODEL_PATH = os.getenv('MODEL_PATH', './saved_models/model_20251007_123355.h5')
    LABEL_PATH = os.getenv('LABEL_PATH', './saved_models/id2label_20251007_123355.pkl')
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
        logger.error(f"❌ Model files not found. MODEL_PATH: {MODEL_PATH}, LABEL_PATH: {LABEL_PATH}")
        # In a real-world scenario, this would raise an exception or use a fallback.
        # For this task, we'll proceed but log the error.
        # For now, we'll assume the files exist or the user will provide them.
        # To avoid startup failure, we'll use dummy paths and rely on the user to ensure correct setup.
        pass

    try:
        model_wrapper = ModelWrapper(
            model_path=MODEL_PATH,
            label_path=LABEL_PATH,
            config=ModelConfig(),
            mongodb_manager=mongodb_manager
        )
        # The ModelWrapper initialization handles the semantic model setup for the input_validator
    except Exception as e:
        logger.error(f"❌ Failed to load model or encoders: {e}")
        # Set to None to prevent further use
        model_wrapper = None

@app.on_event("shutdown")
def shutdown_event():
    global mongodb_manager
    if mongodb_manager:
        mongodb_manager.close()
        logger.info("Service shutdown complete.")

# --- Health Check ---
@app.get("/health")
async def health_check():
    """Check the health status of the service and its components."""
    status = {
        "status": "ok",
        "model_loaded": model_wrapper is not None,
        "mongodb_connected": mongodb_manager.connected if mongodb_manager else False,
        "semantic_check_enabled": input_validator.semantic_check_enabled,
        "timestamp": str(os.getenv('START_TIME', 'N/A'))
    }
    if not status["model_loaded"]:
        status["status"] = "degraded"
    return status

# --- Prediction Endpoints ---
@app.post("/predict", response_model=PredictResponse)
async def predict_with_context(request: PredictWithContextRequest):
    """
    Perform prediction with optional RAG context retrieval.
    """
    if model_wrapper is None:
        raise HTTPException(status_code=503, detail="Model service is not ready or failed to load.")

    text = request.text.strip()
    
    # 1. Input Validation
    is_safe, error_msg = is_safe_input(text)
    if not is_safe:
        raise HTTPException(status_code=400, detail=f"Input validation failed: {error_msg}")

    context_data = None
    context_text = None
    retrieval_status = "success" # Default to success, will be updated if retrieval is attempted

    # Retrieval-Augmented Generation
    if request.use_retrieval:
        try:
            context_data = model_wrapper.retriever.retrieve_context(text, request.sources)
            context_text = context_data.get('context_text')
            
            if not context_text:
                retrieval_status = "empty"
                logger.info("Retrieval returned no context.")
            else:
                logger.info(f"Retrieval successful. Found {context_data.get('total_sources', 0)} sources.")
                
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            retrieval_status = "failed"
            context_data = {"error": str(e)}
            # Continue prediction without context

    # Model Prediction
    try:
        prediction_result = model_wrapper.predict(text, context_text)
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # Save to DB (Optional)
    saved_to_db = False
    db_id = None
    if request.save_to_db and mongodb_manager and mongodb_manager.connected:
        db_id = mongodb_manager.save_prediction(
            text=text,
            label=prediction_result['predicted_label'],
            confidence=prediction_result['confidence'],
            probabilities=prediction_result['probabilities'],
            context=context_data
        )
        saved_to_db = db_id is not None

    # 5. Construct Response
    response_data = {
        "text": text,
        "predicted_label": prediction_result['predicted_label'],
        "confidence": prediction_result['confidence'],
        "probabilities": prediction_result['probabilities'],
        "context": context_data,
        "retrieval_status": retrieval_status,
        "saved_to_db": saved_to_db,
        "db_id": db_id
    }
    
    return PredictResponse(**response_data)

@app.post("/predict-batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """
    Perform batch prediction for a list of texts.
    """
    if model_wrapper is None:
        raise HTTPException(status_code=503, detail="Model service is not ready or failed to load.")

    texts = [t.strip() for t in request.texts]
    
    # Input Validation for all texts
    safe_texts = []
    for text in texts:
        is_safe, error_msg = is_safe_input(text)
        if not is_safe:
            raise HTTPException(status_code=400, detail=f"Batch input validation failed for text '{text[:30]}...': {error_msg}")
        safe_texts.append(text)

    # Model Prediction
    try:
        results = model_wrapper.predict_batch(safe_texts)
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

    return BatchPredictResponse(count=len(results), results=results)

# --- MongoDB Endpoints ---
@app.post("/db/import", response_model=BulkImportResponse)
async def bulk_import(request: BulkImportRequest):
    """Bulk import data into MongoDB."""
    if mongodb_manager is None or not mongodb_manager.connected:
        raise HTTPException(status_code=503, detail="MongoDB service is not connected.")
    
    imported_count = mongodb_manager.import_from_list(request.data)
    
    return BulkImportResponse(
        success=imported_count > 0,
        imported_count=imported_count,
        message=f"Successfully imported {imported_count} records."
    )

@app.get("/db/stats")
async def get_db_stats():
    """Get statistics about the data in the MongoDB collection."""
    if mongodb_manager is None or not mongodb_manager.connected:
        raise HTTPException(status_code=503, detail="MongoDB service is not connected.")
    
    stats = mongodb_manager.get_stats()
    if 'error' in stats:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stats: {stats['error']}")
    
    return stats

@app.get("/db/query")
async def query_db(label: str, limit: int = 10):
    """Query documents by label."""
    if mongodb_manager is None or not mongodb_manager.connected:
        raise HTTPException(status_code=503, detail="MongoDB service is not connected.")
    
    results = mongodb_manager.query_by_label(label, limit)
    
    return {"label": label, "limit": limit, "count": len(results), "results": results}

# --- Main Execution ---
if __name__ == "__main__":
    # Set a start time environment variable for the health check
    os.environ['START_TIME'] = str(datetime.now())
    
    # Check for model paths in command line arguments
    if len(sys.argv) > 2:
        MODEL_PATH = sys.argv[1]
        LABEL_PATH = sys.argv[2]
        os.environ['MODEL_PATH'] = MODEL_PATH
        os.environ['LABEL_PATH'] = LABEL_PATH
        logger.info(f"Using command line arguments for model paths: {MODEL_PATH}, {LABEL_PATH}")
    
    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)