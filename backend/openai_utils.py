"""
OpenAI Utilities Module

Handles Azure OpenAI API interactions, prompt management, and language detection
for the HMO chatbot system. Supports both information collection and QA phases
with RAG implementation for the latter.
"""

# Standard library imports
import os
import re
import logging
from time import time

# Third-party imports
from openai import AzureOpenAI
from fastapi import HTTPException
from dotenv import load_dotenv

# Local imports
from models import ChatRequest
from retriever import retrieve_top_k
from shared.logger_config import logger
from shared.monitoring import monitoring

# Configure base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")

# Load environment variables for Azure OpenAI
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)


def load_prompt(filename: str) -> str:
    """Load prompt template from file
    
    Args:
        filename: Name of prompt template file
        
    Returns:
        String containing prompt template
    """
    with open(os.path.join(PROMPTS_DIR, filename), encoding="utf-8") as f:
        return f.read()

def detect_language(text: str) -> str:
    """Detect primary language of input text
    
    Args:
        text: Input text to analyze
        
    Returns:
        'he' for Hebrew-dominant text, 'en' otherwise
    """
    hebrew_chars = re.findall(r'[\u0590-\u05FF]', text)
    english_chars = re.findall(r'[A-Za-z]', text)
    total = len(hebrew_chars) + len(english_chars)
    return "he" if total and len(hebrew_chars) / total > 0.6 else "en"

async def get_response_from_llm(req: ChatRequest) -> dict:
    """Get response from Azure OpenAI based on request phase"""
    start_time = time()
    
    try:
        if req.phase == "qa":
            # Log RAG request
            logger.info("Starting RAG retrieval",
                phase="qa",
                question_length=len(req.question)
            )
            
            relevant_docs = retrieve_top_k(req.question, k=2)
            
            if relevant_docs[0]["domain"] == "no_match":
                logger.info("No relevant documents found",
                    phase="qa",
                    question=req.question
                )
                return {"answer": relevant_docs[0]["text"]}
            
            # Log successful retrieval
            logger.info("Documents retrieved successfully",
                phase="qa",
                docs_found=len(relevant_docs),
                top_score=relevant_docs[0]["score"]
            )
            
            # Combine retrieved documents for context
            context = "\n\n".join([doc["text"] for doc in relevant_docs])
            
            # Format QA prompt with context
            system_prompt = load_prompt("answer_question.txt").format(
                hmo=req.hmo,
                tier=req.tier,
                context=context
            )
        else:
            logger.info("Collection phase request",
                phase="collection",
                language=req.language
            )
            system_prompt = load_prompt("collect_info.txt")

        # Prepare messages for chat completion
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(req.history)
        
        # Get completion from Azure OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7
        )
        
        duration_ms = int((time() - start_time) * 1000)
        
        # Log LLM metrics
        monitoring.log_llm_call(
            duration_ms=duration_ms,
            success=True,
            phase=req.phase,
            response_length=len(response.choices[0].message.content)
        )
        
        return {"answer": response.choices[0].message.content}
        
    except Exception as e:
        duration_ms = int((time() - start_time) * 1000)
        monitoring.log_llm_call(
            duration_ms=duration_ms,
            success=False,
            phase=req.phase,
            error_type=type(e).__name__
        )
        raise
