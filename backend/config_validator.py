# backend/config_validator.py
"""Configuration validation utilities"""

import os
from config import config
from shared.logger_config import logger

def validate_config() -> bool:
    """Validate all configuration settings"""
    errors = []
    
    # Validate Azure OpenAI config
    if not config.azure_openai.api_key:
        errors.append("AZURE_OPENAI_KEY is not set")
    
    if not config.azure_openai.endpoint:
        errors.append("AZURE_OPENAI_ENDPOINT is not set")
    
    # Validate file paths
    if not os.path.exists(config.prompts_dir):
        errors.append(f"Prompts directory not found: {config.prompts_dir}")
    
    if not os.path.exists(config.embeddings_file):
        errors.append(f"Embeddings file not found: {config.embeddings_file}")
    
    # Validate numeric ranges
    if not 0 <= config.chat.collection_temperature <= 2:
        errors.append("Collection temperature must be between 0 and 2")
    
    if not 0 <= config.chat.qa_temperature <= 2:
        errors.append("QA temperature must be between 0 and 2")
    
    if config.chat.max_tokens <= 0:
        errors.append("max_tokens must be positive")
    
    if errors:
        for error in errors:
            logger.error("Configuration error", error=error)
        return False
    
    logger.info("Configuration validation passed")
    return True