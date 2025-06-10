# backend/config.py
"""
Centralized configuration for the HMO Chatbot system
"""
import os
from dataclasses import dataclass
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI API configuration"""
    api_key: str
    endpoint: str
    api_version: str
    chat_model: str
    embedding_model: str
    
    @classmethod
    def from_env(cls) -> "AzureOpenAIConfig":
        """Create config from environment variables"""
        return cls(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-07-18"),  # Updated API version
            chat_model=os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-4o-mini"),  # Updated default
            embedding_model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        )

@dataclass
class ChatConfig:
    """Chat interaction configuration"""
    collection_temperature: float
    qa_temperature: float
    max_tokens: int
    top_k_documents: int
    similarity_threshold: float
    
    @classmethod
    def default(cls) -> "ChatConfig":
        """Default chat configuration"""
        return cls(
            collection_temperature=0.7,
            qa_temperature=0.7,
            max_tokens=1500,
            top_k_documents=2,
            similarity_threshold=0.75
        )

@dataclass
class SystemConfig:
    """System-wide configuration"""
    azure_openai: AzureOpenAIConfig
    chat: ChatConfig
    
    # File paths
    prompts_dir: str
    embeddings_file: str
    
    # Logging
    log_level: str
    enable_monitoring: bool
    
    @classmethod
    def load(cls) -> "SystemConfig":
        """Load complete system configuration"""
        return cls(
            azure_openai=AzureOpenAIConfig.from_env(),
            chat=ChatConfig.default(),
            prompts_dir=os.path.join(os.path.dirname(__file__), "prompts"),
            embeddings_file="phase2_data/embedded_docs.jsonl",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            enable_monitoring=os.getenv("ENABLE_MONITORING", "true").lower() == "true"
        )

# Global configuration instance
config = SystemConfig.load()