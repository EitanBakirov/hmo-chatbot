import os
from pathlib import Path
from scripts.embed_documents import generate_embeddings
from shared.logger_config import logger
from retriever import load_embeddings


def ensure_embeddings() -> None:
    """Check if embeddings exist, generate if not"""
    embedding_path = Path("phase2_data/embedded_docs.jsonl")
    
    if not embedding_path.exists():
        logger.info("Embeddings file not found, generating embeddings...")
        try:
            generate_embeddings()
            logger.info("Embeddings generated successfully")
        except Exception as e:
            logger.error("Failed to generate embeddings",
                error_type=type(e).__name__,
                error_details=str(e)
            )
            raise


def initialize_backend() -> bool:
    """Initialize all backend components"""
    try:
        # First ensure embeddings exist
        ensure_embeddings()
        
        # Then load them into memory
        load_embeddings()
        
        logger.info("Backend initialization complete",
            status="success",
            components=["embeddings", "logger", "monitoring"]
        )
        return True
    except Exception as e:
        logger.error("Backend initialization failed",
            error=str(e),
            error_type=type(e).__name__
        )
        raise