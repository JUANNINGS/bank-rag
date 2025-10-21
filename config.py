"""
Configuration file for Bank RAG System
Contains Azure OpenAI settings and system parameters

Note: Credentials are read from .env file or environment variables
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load .env file automatically
load_dotenv()

@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI configuration settings"""
    
    # Azure OpenAI Credentials
    # IMPORTANT: Set these via environment variables or update with your values
    api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")  # Your Azure OpenAI API key
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource-name.openai.azure.com/")
    api_version: str = "2024-12-01-preview"
    
    # Deployment Names (update these to match your Azure deployments)
    gpt_deployment: str = os.getenv("AZURE_GPT_DEPLOYMENT", "gpt-4")  # Your GPT model deployment name
    embedding_deployment: str = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")  # Your embedding model deployment name
    
    # Model Parameters
    temperature: float = 0.0  # 0 for deterministic responses
    max_tokens: int = 1000  # Maximum tokens in response
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@dataclass
class RAGConfig:
    """RAG system configuration"""
    
    # Document Processing
    chunk_size: int = 1000  # Characters per chunk
    chunk_overlap: int = 200  # Overlap between chunks
    
    # Retrieval Parameters
    top_k: int = 5  # Number of chunks to retrieve
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
azure_config = AzureOpenAIConfig()
rag_config = RAGConfig()
system_config = SystemConfig()


def get_azure_config() -> AzureOpenAIConfig:
    """Get Azure OpenAI configuration"""
    return azure_config


def get_rag_config() -> RAGConfig:
    """Get RAG configuration"""
    return rag_config


def get_system_config() -> SystemConfig:
    """Get system configuration"""
    return system_config


def update_azure_api_key(api_key: str) -> None:
    """Update Azure OpenAI API key"""
    global azure_config
    azure_config.api_key = api_key


def update_cohere_api_key(api_key: str) -> None:
    """Update Cohere API key for reranking"""
    global system_config
    system_config.cohere_api_key = api_key
