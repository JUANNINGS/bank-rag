"""
RAG Pipeline Module
Main RAG logic with citation and refusal mechanisms
This is the core module that full-stack engineers will integrate with
"""

import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOllama
from retriever import HybridRetriever
from config import get_model_config, get_rag_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """
    Structured response from RAG system
    Designed for easy integration by full-stack team
    """
    answer: str  # Generated answer
    sources: List[Dict[str, str]]  # Source documents used
    confidence: float  # Confidence score (0-1)
    is_refusal: bool  # Whether system refused to answer
    response_time: float  # Time taken in seconds
    metadata: Optional[Dict] = None  # Additional metadata
    confidence_breakdown: Optional[Dict[str, float]] = None  # NEW: Detailed confidence breakdown
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses"""
        return asdict(self)


class BankRAGPipeline:
    """
    Complete RAG pipeline for banking Q&A
    
    Features:
    - Hybrid retrieval (BM25 + Vector + Reranker)
    - Citation tracking (source references)
    - Refusal mechanism (confidence-based)
    - Response time tracking
    """
    
    def __init__(
        self,
        retriever: HybridRetriever,
        refusal_threshold: float = None
    ):
        """
        Initialize RAG pipeline
        
        Args:
            retriever: Hybrid retriever instance
            refusal_threshold: Minimum confidence to answer (default from config)
        """
        model_config = get_model_config()
        rag_config = get_rag_config()
        
        self.retriever = retriever
        self.refusal_threshold = refusal_threshold or rag_config.refusal_threshold
        
        # Initialize Ollama LLM
        self.llm = ChatOllama(
            base_url=model_config.ollama_base_url,
            model=model_config.llm_model,
            temperature=model_config.temperature,
            num_predict=model_config.max_tokens
        )
        
        logger.info(
            f"Initialized RAG Pipeline "
            f"(model={model_config.llm_model}, "
            f"refusal_threshold={self.refusal_threshold})"
        )
    
    def _create_prompt(self, query: str, context_docs: List[Document]) -> str:
        """
        Create prompt for LLM with context and query
        
        Args:
            query: User's question
            context_docs: Retrieved context documents
            
        Returns:
            Formatted prompt string
        """
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            source = doc.metadata.get('source', 'unknown')
            context_parts.append(
                f"[Document {i} - {source}]\n{doc.page_content}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Create prompt with instructions
        prompt = f"""You are a helpful banking assistant for TechBank. Answer the customer's question based on the provided context documents.

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. Be accurate and specific
3. If the context contains the answer, provide it clearly
4. If you're not completely certain, say so
5. Use professional banking language
6. Keep answers concise but complete

CONTEXT DOCUMENTS:
{context}

CUSTOMER QUESTION:
{query}

ANSWER:"""
        
        return prompt
    
    def _calculate_confidence(
        self, 
        query: str, 
        retrieved_docs: List[Document],
        answer: str
    ) -> tuple[float, Dict[str, float]]:
        """
        Calculate confidence score using actual retrieval similarity scores
        
        IMPROVED: Now uses actual document similarity scores instead of simple heuristics
        
        Factors:
        - Best document similarity score (40% weight)
        - Average similarity of top documents (20% weight)
        - Answer quality and specificity (20% weight)
        - Uncertainty phrase detection (20% weight)
        
        Args:
            query: Original query
            retrieved_docs: Retrieved context documents
            answer: Generated answer
            
        Returns:
            Tuple of (confidence_score, breakdown_dict)
        """
        breakdown = {}
        confidence = 0.0
        
        # Factor 1: Document similarity scores (60% weight total)
        # Try to get similarity scores from document metadata
        similarities = []
        for doc in retrieved_docs:
            if hasattr(doc, 'metadata') and 'similarity_score' in doc.metadata:
                similarities.append(doc.metadata['similarity_score'])
        
        if similarities:
            # Use actual similarity scores
            best_sim = max(similarities)
            avg_sim = sum(similarities) / len(similarities)
            
            breakdown['best_similarity'] = best_sim
            breakdown['avg_similarity'] = avg_sim
            
            # Best similarity contribution (40% weight)
            # Scale: 0.9+ → 0.4, 0.8-0.9 → 0.3, 0.7-0.8 → 0.2, <0.7 → 0.1
            if best_sim >= 0.9:
                sim_contribution = 0.4
            elif best_sim >= 0.8:
                sim_contribution = 0.3
            elif best_sim >= 0.7:
                sim_contribution = 0.2
            else:
                sim_contribution = 0.1
            
            confidence += sim_contribution
            breakdown['similarity_contribution'] = sim_contribution
            
            # Average similarity contribution (20% weight)
            # Direct scaling: avg_sim * 0.25 (capped at 0.2)
            avg_contribution = min(0.2, avg_sim * 0.25)
            confidence += avg_contribution
            breakdown['avg_contribution'] = avg_contribution
        else:
            # Fallback to simple heuristic if no similarity scores available
            if len(retrieved_docs) >= 3:
                doc_contribution = 0.3
            elif len(retrieved_docs) >= 1:
                doc_contribution = 0.2
            else:
                doc_contribution = 0.1
            
            confidence += doc_contribution
            breakdown['doc_count_contribution'] = doc_contribution
            breakdown['note'] = 'Using fallback (no similarity scores)'
        
        # Factor 2: Answer quality (20% weight)
        answer_quality = 0.0
        
        # Length and detail
        if len(answer) > 200:
            answer_quality += 0.1
        elif len(answer) > 100:
            answer_quality += 0.05
        
        # Contains specific information (numbers, amounts)
        if any(char.isdigit() for char in answer):
            answer_quality += 0.05
        
        # Detailed answer (word count)
        if len(answer.split()) > 30:
            answer_quality += 0.05
        
        confidence += answer_quality
        breakdown['answer_quality'] = answer_quality
        
        # Factor 3: Uncertainty detection (20% weight - negative)
        uncertainty_penalty = 0.0
        
        uncertainty_phrases = [
            "i'm not sure", "i don't know", "cannot find",
            "no information", "unclear", "uncertain",
            "might be", "possibly", "perhaps"
        ]
        answer_lower = answer.lower()
        
        for phrase in uncertainty_phrases:
            if phrase in answer_lower:
                uncertainty_penalty += 0.1
        
        confidence -= uncertainty_penalty
        breakdown['uncertainty_penalty'] = -uncertainty_penalty
        
        # Factor 4: Refusal detection
        refusal_phrases = [
            "i cannot answer", "don't have enough information",
            "not in the provided context", "cannot determine"
        ]
        
        has_refusal = any(phrase in answer_lower for phrase in refusal_phrases)
        if has_refusal:
            confidence -= 0.3
            breakdown['refusal_penalty'] = -0.3
        else:
            breakdown['refusal_penalty'] = 0.0
        
        # Clamp between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        breakdown['final_confidence'] = confidence
        
        return confidence, breakdown
    
    def _extract_sources(self, retrieved_docs: List[Document]) -> List[Dict[str, str]]:
        """
        Extract source information from retrieved documents
        
        Args:
            retrieved_docs: Retrieved context documents
            
        Returns:
            List of source dictionaries
        """
        sources = []
        seen_sources = set()
        
        for doc in retrieved_docs:
            source_name = doc.metadata.get('source', 'unknown')
            
            # Avoid duplicate sources
            if source_name in seen_sources:
                continue
            seen_sources.add(source_name)
            
            sources.append({
                'source': source_name,
                'chunk_id': doc.metadata.get('chunk_id', 'unknown'),
                'excerpt': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        
        return sources
    
    def query(
        self, 
        question: str,
        include_sources: bool = True,
        force_answer: bool = False
    ) -> RAGResponse:
        """
        Main query function - this is what full-stack team will call
        
        Args:
            question: User's question
            include_sources: Whether to include source citations
            force_answer: Force answer even below confidence threshold
            
        Returns:
            RAGResponse object with answer, sources, and metadata
        """
        start_time = time.time()
        
        logger.info(f"Processing query: '{question}'")
        
        try:
            # Step 1: Retrieve relevant documents with similarity scores
            logger.debug("Retrieving relevant documents...")
            retrieved_docs = self.retriever.invoke(question)
            
            # Try to add similarity scores to documents for better confidence calculation
            try:
                if hasattr(self.retriever, 'vector_retriever'):
                    vector_store = self.retriever.vector_retriever.vectorstore
                    docs_with_scores = vector_store.similarity_search_with_score(question, k=5)
                    
                    # Add similarity scores to metadata
                    # FAISS returns L2 distance, convert to similarity: similarity ≈ 1 / (1 + distance)
                    for i, (doc, distance) in enumerate(docs_with_scores):
                        if i < len(retrieved_docs):
                            similarity = 1.0 / (1.0 + distance)
                            retrieved_docs[i].metadata['similarity_score'] = similarity
            except Exception as e:
                logger.debug(f"Could not get similarity scores: {e}")
                # Continue without similarity scores (will use fallback)
            
            if not retrieved_docs:
                logger.warning("No documents retrieved")
                return RAGResponse(
                    answer="I couldn't find relevant information to answer your question. Please try rephrasing or contact customer support.",
                    sources=[],
                    confidence=0.0,
                    is_refusal=True,
                    response_time=time.time() - start_time,
                    metadata={'reason': 'no_documents_retrieved'}
                )
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            # Step 2: Create prompt with context
            prompt = self._create_prompt(question, retrieved_docs)
            
            # Step 3: Generate answer using LLM
            logger.debug("Generating answer...")
            response = self.llm.invoke(prompt)
            answer = response.content.strip()
            
            # Step 4: Calculate confidence with breakdown
            confidence, confidence_breakdown = self._calculate_confidence(
                question, 
                retrieved_docs, 
                answer
            )
            logger.info(f"Confidence score: {confidence:.2%}")
            logger.debug(f"Confidence breakdown: {confidence_breakdown}")
            
            # Step 5: Refusal mechanism
            is_refusal = False
            if confidence < self.refusal_threshold and not force_answer:
                logger.info(f"Confidence {confidence:.2f} below threshold {self.refusal_threshold}, refusing to answer")
                is_refusal = True
                answer = (
                    "I don't have enough confidence in my answer based on the available information. "
                    "For accurate information, please:\n"
                    "- Contact customer service at 1-800-TECHBANK\n"
                    "- Visit a TechBank branch\n"
                    "- Check our website at techbank.com\n"
                )
            
            # Step 6: Extract sources (citations)
            sources = []
            if include_sources and not is_refusal:
                sources = self._extract_sources(retrieved_docs)
            
            # Calculate response time
            response_time = time.time() - start_time
            logger.info(f"Response generated in {response_time:.2f} seconds")
            
            # Return structured response
            return RAGResponse(
                answer=answer,
                sources=sources,
                confidence=confidence,
                is_refusal=is_refusal,
                response_time=response_time,
                metadata={
                    'num_retrieved_docs': len(retrieved_docs),
                    'question_length': len(question),
                    'answer_length': len(answer)
                },
                confidence_breakdown=confidence_breakdown
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return RAGResponse(
                answer=f"An error occurred while processing your question. Please try again or contact support.",
                sources=[],
                confidence=0.0,
                is_refusal=True,
                response_time=time.time() - start_time,
                metadata={'error': str(e)}
            )


def create_rag_pipeline(retriever: HybridRetriever) -> BankRAGPipeline:
    """
    Convenience function to create RAG pipeline
    
    Args:
        retriever: Configured hybrid retriever
        
    Returns:
        BankRAGPipeline instance
    """
    return BankRAGPipeline(retriever)


# Example usage
if __name__ == "__main__":
    from document_loader import load_banking_documents
    from chunking import chunk_banking_documents
    from embeddings import create_embeddings_generator
    from retriever import create_hybrid_retriever
    
    print("="*60)
    print("Testing Bank RAG Pipeline")
    print("="*60)
    
    try:
        # Setup pipeline
        print("\n1. Setting up RAG pipeline...")
        documents = load_banking_documents("./documents")
        chunks = chunk_banking_documents(documents)
        embeddings_gen = create_embeddings_generator()
        embeddings_model = embeddings_gen.get_embeddings_model()
        retriever = create_hybrid_retriever(chunks, embeddings_model)
        pipeline = create_rag_pipeline(retriever)
        print("✓ Pipeline ready")
        
        # Test queries
        test_queries = [
            "What are the fees for overdraft protection?",
            "How do I apply for a personal loan?",
            "What is the interest rate on savings accounts?"
        ]
        
        print("\n2. Testing queries...")
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Query {i} ---")
            print(f"Q: {query}")
            
            response = pipeline.query(query)
            
            print(f"A: {response.answer[:200]}...")
            print(f"Confidence: {response.confidence:.2f}")
            print(f"Sources: {len(response.sources)}")
            print(f"Time: {response.response_time:.2f}s")
            print(f"Refusal: {response.is_refusal}")
            
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
