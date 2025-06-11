"""
FastAPI Backend Server

Main entry point for the HMO chatbot backend service. Handles:
- REST API endpoints
- CORS configuration
- Request/response processing
- Error handling and logging

The server provides a single endpoint '/ask' that processes both:
- Information collection phase
- Question answering phase using RAG
"""

# FastAPI and middleware imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Local application imports
from models import ChatRequest, ChatResponse
from openai_utils import get_response_from_llm
from initialize import initialize_backend
from shared.logger_config import logger
from config import config
from config_validator import validate_config
# Monitoring utilities
from shared import monitoring

# Standard library imports
from time import time
from datetime import datetime

# Initialize FastAPI application
app = FastAPI(
    title="HMO Chatbot API",
    description="Backend API for HMO service inquiries",
    version="1.0.0"
)

# Configure CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Run initialization tasks with config validation"""
    logger.info("Starting HMO Chatbot API",
        version="1.0.0",
        environment="production",
        chat_model=config.azure_openai.chat_model,
        api_version=config.azure_openai.api_version
    )
    
    # Validate configuration
    if not validate_config():
        raise RuntimeError("Configuration validation failed")
    
    # Run complete backend initialization
    initialize_backend()

@app.post("/ask", response_model=ChatResponse)
async def ask(req: ChatRequest):
    """Process chat requests and return responses"""
    start_time = time()
    
    try:
        result = await get_response_from_llm(req) 
        duration_ms = int((time() - start_time) * 1000)
        
        logger.info("Request processed successfully",
            duration_ms=duration_ms,
            phase=req.phase
        )
        return ChatResponse(**result)
        
    except Exception as e:
        duration_ms = int((time() - start_time) * 1000)
        logger.error("Request processing failed",
            error_type=type(e).__name__,
            error_details=str(e),
            duration_ms=duration_ms,
            phase=req.phase
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Log health check with basic metrics
        logger.info("Health check performed",
            status="healthy",
            timestamp=int(time())
        )
        return {"status": "healthy"}
    except Exception as e:
        logger.error("Health check failed",
            error_type=type(e).__name__,
            error_details=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail="Health check failed"
        )

@app.get("/metrics")
async def get_metrics():
    """Return current monitoring metrics"""
    return {
        "metrics": monitoring.metrics,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics/reset")
async def reset_metrics():
    """Reset monitoring metrics"""
    monitoring.metrics = {  # Reset the existing instance instead of creating new
        "llm_calls": {
            "success": 0,
            "failed": 0,
            "total_time_ms": 0,
            "average_time_ms": 0
        },
        "rag_queries": {
            "total": 0,
            "no_matches": 0,
            "average_similarity": 0.0
        },
        "conversation": {
            "collection_phase": {"success": 0, "failed": 0},
            "qa_phase": {"success": 0, "failed": 0},
            "language_stats": {"he": 0, "en": 0}
        }
    }
    return {"status": "Metrics reset successfully"}