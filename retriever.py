"""
Hybrid Retriever Module
Implements BM25 + Vector Search + Reranking following RAG.ipynb patterns
"""

import logging
from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from config import get_rag_config, get_system_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining:
    1. BM25 (keyword-based search)
    2. Vector similarity (semantic search)
    3. Optional reranking (Cohere)
    
    Following the architecture from RAG.ipynb
    """
    
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        reranker: Optional[Any] = None,
        top_k: int = None
    ):
        """
        Initialize hybrid retriever
        
        Args:
            vector_retriever: FAISS or other vector retriever
            bm25_retriever: BM25 keyword retriever
            reranker: Optional reranker (Cohere)
            top_k: Number of results to return
        """
        config = get_rag_config()
        
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.top_k = top_k or config.top_k
        
        logger.info(
            f"Initialized HybridRetriever (top_k={self.top_k}, "
            f"reranker={'enabled' if reranker else 'disabled'})"
        )
    
    def invoke(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents using hybrid approach
        
        Process:
        1. Get results from vector retriever (semantic)
        2. Get results from BM25 retriever (keyword)
        3. Combine and deduplicate results
        4. Optionally rerank using Cohere
        5. Return top-k results
        
        Args:
            query: User's question
            
        Returns:
            List of relevant Document objects with scores
        """
        logger.info(f"Processing query: '{query}'")
        
        # Get results from both retrievers
        logger.debug("Retrieving from vector store...")
        vector_docs = self.vector_retriever.invoke(query)
        
        logger.debug("Retrieving from BM25...")
        bm25_docs = self.bm25_retriever.invoke(query)
        
        logger.info(
            f"Retrieved {len(vector_docs)} vector docs, "
            f"{len(bm25_docs)} BM25 docs"
        )
        
        # Combine results, removing duplicates
        # Keep docs from vector search first (typically better quality)
        combined_docs = vector_docs.copy()
        
        # Add BM25 docs that aren't already included
        seen_content = {doc.page_content for doc in combined_docs}
        for doc in bm25_docs:
            if doc.page_content not in seen_content:
                combined_docs.append(doc)
                seen_content.add(doc.page_content)
        
        logger.info(f"Combined to {len(combined_docs)} unique documents")
        
        # Rerank if reranker is available
        if self.reranker and combined_docs:
            logger.debug("Reranking results...")
            try:
                reranked_docs = self.reranker.compress_documents(
                    combined_docs, 
                    query
                )
                logger.info(f"Reranked to {len(reranked_docs)} documents")
                return reranked_docs
            except Exception as e:
                logger.warning(f"Reranking failed: {str(e)}, returning combined results")
                return combined_docs[:self.top_k]
        
        # Return top-k combined results (no reranking)
        return combined_docs[:self.top_k]


def create_bm25_retriever(chunks: List[Document], k: int = None) -> BM25Retriever:
    """
    Create BM25 retriever for keyword-based search
    
    Args:
        chunks: List of document chunks
        k: Number of results to return (default from config)
        
    Returns:
        BM25Retriever instance
    """
    config = get_rag_config()
    k = k or config.top_k
    
    logger.info(f"Creating BM25 retriever with k={k}")
    retriever = BM25Retriever.from_documents(chunks, language="english")
    retriever.k = k
    
    return retriever


def create_vector_retriever(
    chunks: List[Document], 
    embeddings_model,
    k: int = None,
    save_path: Optional[str] = None
) -> tuple:
    """
    Create FAISS vector retriever for semantic search
    
    Args:
        chunks: List of document chunks
        embeddings_model: Embeddings instance (HuggingFaceEmbeddings)
        k: Number of results to return (default from config)
        save_path: Optional path to save FAISS index
        
    Returns:
        Tuple of (retriever, vector_store)
    """
    config = get_rag_config()
    k = k or config.top_k
    
    logger.info(f"Creating FAISS vector retriever with k={k}")
    logger.info(f"Embedding {len(chunks)} chunks (this may take a minute)...")
    
    # Create FAISS vector store
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    
    # Save index if path provided
    if save_path:
        logger.info(f"Saving FAISS index to {save_path}")
        vector_store.save_local(save_path)
    
    # Create retriever from vector store
    retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )
    
    logger.info(f"✓ Vector store created with {len(chunks)} chunks")
    
    return retriever, vector_store


def create_reranker(top_n: int = None) -> Optional[Any]:
    """
    Create Cohere reranker (optional)
    
    Args:
        top_n: Number of documents to return after reranking
        
    Returns:
        Cohere reranker or None if API key not available
    """
    sys_config = get_system_config()
    rag_config = get_rag_config()
    top_n = top_n or rag_config.reranker_top_n
    
    if not sys_config.cohere_api_key:
        logger.warning("Cohere API key not set, reranking disabled")
        return None
    
    try:
        from langchain_cohere import CohereRerank
        
        logger.info(f"Creating Cohere reranker (top_n={top_n})")
        reranker = CohereRerank(
            cohere_api_key=sys_config.cohere_api_key,
            top_n=top_n,
            model=sys_config.cohere_model
        )
        return reranker
        
    except ImportError:
        logger.warning("langchain-cohere not installed, reranking disabled")
        return None
    except Exception as e:
        logger.warning(f"Failed to create reranker: {str(e)}")
        return None


def create_hybrid_retriever(
    chunks: List[Document],
    embeddings_model,
    use_reranker: bool = False,
    save_vector_store: bool = True
) -> HybridRetriever:
    """
    Create complete hybrid retriever system
    
    Args:
        chunks: List of document chunks
        embeddings_model: Embeddings instance (HuggingFaceEmbeddings)
        use_reranker: Whether to use Cohere reranker
        save_vector_store: Whether to save FAISS index
        
    Returns:
        HybridRetriever instance
    """
    config = get_rag_config()
    
    logger.info("="*60)
    logger.info("Creating Hybrid Retrieval System")
    logger.info("="*60)
    
    # Create vector retriever
    logger.info("\n1. Creating vector retriever...")
    save_path = config.vector_store_path if save_vector_store else None
    vector_retriever, _ = create_vector_retriever(
        chunks,
        embeddings_model,
        save_path=save_path
    )
    
    # Create BM25 retriever
    logger.info("\n2. Creating BM25 retriever...")
    bm25_retriever = create_bm25_retriever(chunks)
    
    # Create reranker (optional)
    reranker = None
    if use_reranker:
        logger.info("\n3. Creating reranker...")
        reranker = create_reranker()
        if reranker:
            logger.info("✓ Reranker enabled")
        else:
            logger.info("✗ Reranker not available")
    else:
        logger.info("\n3. Skipping reranker (disabled)")
    
    # Create hybrid retriever
    logger.info("\n4. Assembling hybrid retriever...")
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker
    )
    
    logger.info("\n" + "="*60)
    logger.info("✓ Hybrid Retrieval System Ready")
    logger.info("="*60)
    
    return hybrid_retriever


# Example usage
if __name__ == "__main__":
    from document_loader import load_banking_documents
    from chunking import chunk_banking_documents
    from embeddings import create_embeddings_generator
    
    print("="*60)
    print("Testing Hybrid Retriever")
    print("="*60)
    
    try:
        # 1. Load documents
        print("\n1. Loading documents...")
        documents = load_banking_documents("./documents")
        print(f"✓ Loaded {len(documents)} document(s)")
        
        # 2. Chunk documents
        print("\n2. Chunking documents...")
        chunks = chunk_banking_documents(documents)
        print(f"✓ Created {len(chunks)} chunk(s)")
        
        # 3. Create embeddings generator
        print("\n3. Initializing embeddings...")
        embeddings_gen = create_embeddings_generator()
        embeddings_model = embeddings_gen.get_embeddings_model()
        print("✓ Embeddings ready")
        
        # 4. Create hybrid retriever
        print("\n4. Building hybrid retriever...")
        retriever = create_hybrid_retriever(
            chunks,
            embeddings_model,
            use_reranker=False,  # Set to True if you have Cohere API key
            save_vector_store=True
        )
        
        # 5. Test retrieval
        print("\n5. Testing retrieval...")
        test_query = "What are the fees for wire transfers?"
        results = retriever.invoke(test_query)
        
        print(f"\n✓ Retrieved {len(results)} results for: '{test_query}'")
        print("\nTop result:")
        print(f"  Source: {results[0].metadata.get('source', 'unknown')}")
        print(f"  Content preview: {results[0].page_content[:200]}...")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
