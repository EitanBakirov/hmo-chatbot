"""
OpenAI Utilities Module with centralized configuration

Handles Azure OpenAI API interactions, prompt management, and language detection
for the HMO chatbot system. Supports both information collection and QA phases
with RAG implementation for the latter.
"""

# Standard library imports
import os
import re
import logging
from time import time
import json
from typing import Dict, List

# Third-party imports
import numpy as np
from openai import AzureOpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam, 
    ChatCompletionAssistantMessageParam
)

# Local imports
from models import ChatRequest
from retriever import retrieve_top_k
from shared.logger_config import logger
from function_definitions import COLLECTION_FUNCTIONS
from config import config  # Import centralized config

# Initialize Azure OpenAI client using config
client = AzureOpenAI(
    api_key=config.azure_openai.api_key,
    api_version=config.azure_openai.api_version,
    azure_endpoint=config.azure_openai.endpoint
)

def load_prompt(filename: str) -> str:
    """Load system prompt from file using config path
    
    Args:
        filename: Name of prompt template file
        
    Returns:
        String containing prompt template
    """
    try:
        filepath = os.path.join(config.prompts_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.error("Prompt file not found", filename=filename)
        raise

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

def get_response_from_llm(req: ChatRequest) -> dict:
    """Get response from Azure OpenAI based on request phase"""
    start_time = time()
    
    try:
        if req.phase == "collection":
            return handle_collection_phase(req)
        else:
            return handle_qa_phase(req)
    except Exception as e:
        logger.error("LLM request failed",
            error_type=type(e).__name__,
            error_details=str(e),
            phase=req.phase
        )
        raise

def handle_collection_phase(req: ChatRequest) -> dict:
    """Handle data collection phase with centralized config"""
    logger.info("Collection phase request",
        phase="collection",
        language=req.language,
        question_length=len(req.question)
    )
    
    system_prompt = load_prompt("collect_info.txt")
    messages = prepare_messages(system_prompt, req.history, req.question)
    
    response = client.chat.completions.create(
        model=config.azure_openai.chat_model,           
        messages=messages,
        temperature=config.chat.collection_temperature,  
        max_tokens=config.chat.max_tokens,              
        functions=COLLECTION_FUNCTIONS,
        function_call="auto"
    )
    
    message = response.choices[0].message
    
    # Check if function was called (phase transition)
    if message.function_call and message.function_call.name == "complete_data_collection":
        try:
            user_info = json.loads(message.function_call.arguments)
            
            logger.info("Phase transition triggered",
                from_phase="collection",
                to_phase="qa",
                collected_fields=list(user_info.keys())
            )
            
            success_message = generate_success_message(user_info.get("preferred_language", "he"))
            
            return {
                "answer": message.content or success_message,
                "phase_transition": True,
                "user_info": user_info
            }
            
        except json.JSONDecodeError as e:
            logger.error("Function call parsing failed",
                error=str(e),
                raw_arguments=message.function_call.arguments
            )
            return {
                "answer": "אירעה שגיאה בעיבוד המידע. אנא נסה שוב.",
                "phase_transition": False
            }
    
    # Regular collection conversation
    return {
        "answer": message.content,
        "phase_transition": False
    }

def handle_qa_phase(req: ChatRequest) -> dict:
    """Handle QA phase with centralized config"""
    logger.info("Starting RAG retrieval",
        phase="qa",
        question_length=len(req.question)
    )
    
    # Use config for retrieval parameters
    relevant_docs = retrieve_top_k(
        req.question, 
        k=config.chat.top_k_documents 
    )
    
    if relevant_docs[0]["domain"] == "no_match":
        return {"answer": relevant_docs[0]["text"]}
    
    context = "\n\n".join([doc["text"] for doc in relevant_docs])
    system_prompt = load_prompt("answer_question.txt").format(
        hmo=req.hmo,
        tier=req.tier,
        context=context
    )
    
    messages = prepare_messages(system_prompt, req.history, req.question)
    
    response = client.chat.completions.create(
        model=config.azure_openai.chat_model,      
        messages=messages,
        temperature=config.chat.qa_temperature,    
        max_tokens=config.chat.max_tokens          
        # No functions parameter for QA phase
    )
    
    return {
        "answer": response.choices[0].message.content,
        "phase_transition": False
    }

def generate_success_message(language: str) -> str:
    """Generate success message for completed data collection"""
    if language == "en":
        return "Thank you! Your information has been collected successfully. I can now help you with questions about your health fund benefits."
    else:
        return "תודה רבה! המידע שלך נקלט בהצלחה. כעת אני יכול לעזור לך עם שאלות לגבי הזכויות שלך בקופת החולים."

def prepare_messages(system_prompt: str, history: list, current_question: str) -> list:
    """Prepare messages for OpenAI chat completion"""
    messages = [ChatCompletionSystemMessageParam(role="system", content=system_prompt)]
    
    for msg in history:
        if msg["role"] == "user":
            messages.append(ChatCompletionUserMessageParam(role="user", content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=msg["content"]))
    
    messages.append(ChatCompletionUserMessageParam(role="user", content=current_question))
    
    return messages
