"""
Embeddings Module
Handles text embeddings generation using Azure OpenAI
Following patterns from RAG.ipynb
"""

import logging
from typing import List
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from config import get_azure_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BankEmbeddingsGenerator:
    """
    Embeddings generator using Azure OpenAI
    Converts text chunks into vector embeddings for semantic search
    """
    
    def __init__(self):
        """
        Initialize embeddings generator with Azure OpenAI configuration
        """
        config = get_azure_config()
        
        # Initialize Azure OpenAI Embeddings
        # Uses text-embedding-ada-002 model through Azure deployment
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=config.embedding_deployment,
            openai_api_version=config.api_version,
            azure_endpoint=config.endpoint,
            api_key=config.api_key,
        )
        
        logger.info(
            f"Initialized embeddings with Azure deployment: {config.embedding_deployment}"
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text strings
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each is a list of 1536 floats)
        """
        if not texts:
            logger.warning("No texts provided for embedding")
            return []
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} text(s)...")
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"✓ Generated {len(embeddings)} embedding(s)")
            
            # Log embedding dimensions
            if embeddings:
                logger.info(f"Embedding dimensions: {len(embeddings[0])}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text
        Optimized for query embedding (vs document embedding)
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector (list of 1536 floats)
        """
        try:
            embedding = self.embeddings.embed_query(text)
            logger.debug(f"Generated query embedding (dim={len(embedding)})")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise
    
    def get_embeddings_model(self) -> AzureOpenAIEmbeddings:
        """
        Get the underlying embeddings model
        Useful for passing to vector stores (FAISS, Chroma, etc.)
        
        Returns:
            Azure OpenAI Embeddings model instance
        """
        return self.embeddings


def create_embeddings_generator() -> BankEmbeddingsGenerator:
    """
    Convenience function to create embeddings generator
    
    Returns:
        BankEmbeddingsGenerator instance
    """
    return BankEmbeddingsGenerator()


def embed_document_chunks(chunks: List[Document]) -> tuple:
    """
    Generate embeddings for document chunks
    
    Args:
        chunks: List of Document objects (chunks)
        
    Returns:
        Tuple of (chunks, embeddings_generator)
    """
    generator = create_embeddings_generator()
    
    # Extract text content from chunks
    texts = [chunk.page_content for chunk in chunks]
    
    logger.info(f"Preparing to embed {len(texts)} chunk(s)...")
    
    # Note: Actual embedding generation happens when chunks are added to vector store
    # This function just prepares the generator
    
    return chunks, generator


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Testing Bank Embeddings Generator")
    print("="*60)
    
    try:
        # Initialize generator
        print("\n1. Initializing embeddings generator...")
        generator = create_embeddings_generator()
        print("✓ Embeddings generator initialized")
        
        # Test with sample banking text
        print("\n2. Testing with sample text...")
        sample_texts = [
            "What are the interest rates for personal loans?",
            "How do I apply for a credit card?",
            "Where can I find the nearest ATM?"
        ]
        
        embeddings = generator.embed_documents(sample_texts)
        print(f"✓ Generated {len(embeddings)} embedding(s)")
        print(f"  Embedding dimensions: {len(embeddings[0])}")
        
        # Test query embedding
        print("\n3. Testing query embedding...")
        query = "How much is the overdraft fee?"
        query_embedding = generator.embed_query(query)
        print(f"✓ Generated query embedding (dim={len(query_embedding)})")
        
        # Show similarity (cosine similarity between first doc and query)
        import numpy as np
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        similarity = cosine_similarity(embeddings[0], query_embedding)
        print(f"\n4. Similarity between first text and query: {similarity:.4f}")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
