"""
Retriever Module

Implements RAG (Retrieval Augmented Generation) functionality for the HMO chatbot.
Handles document embedding, similarity search, and retrieval of relevant context
for user queries.

Key Components:
- Document embedding using Azure OpenAI
- Cosine similarity calculation
- Top-k document retrieval
- Relevance thresholding
"""

# Standard library imports
import json
import os
from typing import List, Dict

# Third-party imports
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
from shared.monitoring import monitoring
from shared.logger_config import logger

# Local application imports
from config import config  # Import centralized config

# Configure Azure OpenAI
load_dotenv()

# Constants
MIN_SIMILARITY_THRESHOLD = 0.7  # Minimum score for document relevance
NO_MATCH_MESSAGE = "לא נמצא מידע רלוונטי לשאלה זו. אנא נסח את השאלה מחדש או שאל על נושא אחר."

# Initialize Azure OpenAI client for embeddings using config
client = AzureOpenAI(
    api_key=config.azure_openai.api_key,
    api_version=config.azure_openai.api_version,
    azure_endpoint=config.azure_openai.endpoint
)

# Move global embedded_docs into module scope
embedded_docs = []

def load_embeddings():
    """Load pre-computed document embeddings using config path"""
    global embedded_docs
    try:
        with open(config.embeddings_file, encoding="utf-8") as f: 
            embedded_docs = [json.loads(line) for line in f]
        logger.info("Loaded embeddings successfully",
            count=len(embedded_docs),
            file=config.embeddings_file
        )
    except Exception as e:
        logger.error("Failed to load embeddings",
            error=str(e),
            error_type=type(e).__name__
        )
        raise

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two embedding vectors
    
    Args:
        a: First embedding vector
        b: Second embedding vector
        
    Returns:
        float: Similarity score between 0 (different) and 1 (identical)
    """
    a, b = np.array(a), np.array(b)  # Ensure numpy arrays
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def embed_text(text: str) -> List[float]:
    """Generate embeddings using config model
    
    Args:
        text: Input text to embed
        
    Returns:
        List[float]: Embedding vector from Azure OpenAI
    """
    try:
        response = client.embeddings.create(
            model=config.azure_openai.embedding_model,  
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error("Embedding generation failed", error=str(e))
        raise

def retrieve_top_k(query: str, k: int = None) -> List[Dict]:
    """Retrieve top-k documents using config defaults"""
    if k is None:
        k = config.chat.top_k_documents  # Use config default
    
    try:
        # Generate query embedding
        query_vec = embed_text(query)
        
        # Score all documents
        scored_docs = [
            {
                "domain": doc["domain"],
                "text": doc["text"],
                "score": cosine_similarity(np.array(query_vec), np.array(doc["embedding"]))
            }
            for doc in embedded_docs
        ]
        
        # Get max similarity score
        max_score = max([doc["score"] for doc in scored_docs])
        
        # Filter by relevance
        relevant_docs = [
            doc for doc in scored_docs 
            if doc["score"] >= MIN_SIMILARITY_THRESHOLD
        ]
        
        # Log RAG metrics
        monitoring.log_rag_query(
            similarity_score=max_score,
            found_match=len(relevant_docs) > 0,
            query_length=len(query),
            matched_domains=[d["domain"] for d in relevant_docs[:k]]
        )
        
        # Return results
        if not relevant_docs:
            return [{
                "domain": "no_match",
                "text": NO_MATCH_MESSAGE,
                "score": 0
            }]
        
        return sorted(relevant_docs, key=lambda x: x["score"], reverse=True)[:k]
        
    except Exception as e:
        logger.error("RAG retrieval failed", error_type=type(e).__name__)
        raise
