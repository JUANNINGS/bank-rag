"""
Configuration file for Bank RAG System
Uses Open Source models (Ollama + sentence-transformers)

Note: Credentials are read from .env file or environment variables
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load .env file automatically
load_dotenv()

@dataclass
class ModelConfig:
    """Open Source Model Configuration (Ollama + sentence-transformers)"""
    
    # Ollama Configuration
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    # Lightweight options: phi3:mini (3.8B, ~4GB RAM), llama3.1:8b (8GB RAM)
    # Note: llama3.1:3b does not exist. Use phi3:mini for lightweight option.
    llm_model: str = os.getenv("OLLAMA_LLM_MODEL", "phi3:mini")  # Default: lightweight 3.8B model
    
    # Sentence Transformers Configuration
    # all-MiniLM-L6-v2 is the lightest (384 dims, ~90MB)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # Options: all-MiniLM-L6-v2, multilingual-e5-base
    
    # Model Parameters
    temperature: float = 0.0  # 0 for deterministic responses
    max_tokens: int = 500  # Maximum tokens in response (reduced for performance)
    top_p: float = 1.0


@dataclass
class RAGConfig:
    """RAG system configuration"""
    
    # Document Processing
    chunk_size: int = 500  # Characters per chunk (reduced for performance)
    chunk_overlap: int = 100  # Overlap between chunks (reduced for performance)
    
    # Retrieval Parameters
    top_k: int = 3  # Number of chunks to retrieve (reduced for performance)
    reranker_top_n: int = 3  # Number of chunks after reranking
    
    # Confidence Thresholds
    min_confidence_score: float = 0.5  # Minimum score for confident answer
    refusal_threshold: float = 0.3  # Below this, refuse to answer
    
    # Vector Store
    vector_store_path: str = "./data/vector_store"
    
    # Evaluation
    evaluation_metrics: list = None
    
    def __post_init__(self):
        if self.evaluation_metrics is None:
            self.evaluation_metrics = ["hit_rate", "mrr", "ndcg", "precision"]


@dataclass
class SystemConfig:
    """System-wide configuration"""
    
    # Paths
    documents_path: str = "./documents"
    data_path: str = "./data"
    tests_path: str = "./tests"
    logs_path: str = "./logs"
    
    # Cohere Settings (for reranking)
    cohere_api_key: Optional[str] = None  # Set if using Cohere reranker
    cohere_model: str = "rerank-english-v2.0"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Performance
    batch_size: int = 10  # For batch processing
    max_concurrent_requests: int = 5
    
    # Response Time Target
    target_response_time: float = 3.0  # seconds (from PRD requirement)


# Global configuration instances
model_config = ModelConfig()
rag_config = RAGConfig()
system_config = SystemConfig()


def get_model_config() -> ModelConfig:
    """Get model configuration"""
    return model_config


def get_rag_config() -> RAGConfig:
    """Get RAG configuration"""
    return rag_config


def get_system_config() -> SystemConfig:
    """Get system configuration"""
    return system_config


def update_cohere_api_key(api_key: str) -> None:
    """Update Cohere API key for reranking"""
    global system_config
    system_config.cohere_api_key = api_key
