"""
Main Entry Point for Bank RAG System
Demonstrates setup and usage for full-stack team integration
"""

import logging
import json
from typing import Optional
from document_loader import load_banking_documents
from chunking import chunk_banking_documents
from embeddings import create_embeddings_generator
from retriever import create_hybrid_retriever
from rag_pipeline import create_rag_pipeline, RAGResponse
from evaluation import RAGEvaluator, generate_test_queries
from config import get_system_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BankRAGSystem:
    """
    Complete Bank RAG System
    
    This class provides a simple interface for the full-stack team
    to integrate the RAG system into their application.
    """
    
    def __init__(self, documents_path: str = "./documents"):
        """
        Initialize the complete RAG system
        
        Args:
            documents_path: Path to banking documents directory
        """
        self.documents_path = documents_path
        self.pipeline = None
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Internal method to set up the RAG pipeline"""
        logger.info("="*60)
        logger.info("Initializing Bank RAG System")
        logger.info("="*60)
        
        # Step 1: Load documents
        logger.info("\n1. Loading banking documents...")
        documents = load_banking_documents(self.documents_path)
        logger.info(f"✓ Loaded {len(documents)} document(s)")
        
        # Step 2: Chunk documents
        logger.info("\n2. Chunking documents...")
        chunks = chunk_banking_documents(documents)
        logger.info(f"✓ Created {len(chunks)} chunk(s)")
        
        # Step 3: Create embeddings
        logger.info("\n3. Initializing Azure OpenAI embeddings...")
        embeddings_gen = create_embeddings_generator()
        embeddings_model = embeddings_gen.get_embeddings_model()
        logger.info("✓ Embeddings ready")
        
        # Step 4: Create retriever
        logger.info("\n4. Building hybrid retriever (BM25 + Vector)...")
        retriever = create_hybrid_retriever(
            chunks,
            embeddings_model,
            use_reranker=False,  # Set to True if Cohere API key available
            save_vector_store=True
        )
        logger.info("✓ Retriever ready")
        
        # Step 5: Create RAG pipeline
        logger.info("\n5. Initializing RAG pipeline...")
        self.pipeline = create_rag_pipeline(retriever)
        logger.info("✓ Pipeline ready")
        
        logger.info("\n" + "="*60)
        logger.info("✓ Bank RAG System Ready")
        logger.info("="*60 + "\n")
    
    def query(self, question: str, **kwargs) -> RAGResponse:
        """
        Query the RAG system
        
        Args:
            question: User's question
            **kwargs: Additional parameters (include_sources, force_answer)
            
        Returns:
            RAGResponse object
        """
        return self.pipeline.query(question, **kwargs)
    
    def batch_query(self, questions: list) -> list:
        """
        Process multiple questions
        
        Args:
            questions: List of questions
            
        Returns:
            List of RAGResponse objects
        """
        return [self.query(q) for q in questions]


def demo_interactive():
    """Interactive demo for testing the system"""
    print("\n" + "="*60)
    print("Bank RAG System - Interactive Demo")
    print("="*60)
    print("\nInitializing system (this may take a minute)...")
    
    # Initialize system
    system = BankRAGSystem()
    
    print("\n✓ System ready! Ask questions about TechBank services.")
    print("Type 'quit' to exit, 'examples' for sample questions.\n")
    
    example_questions = [
        "What are the fees for wire transfers?",
        "How do I apply for a personal loan?",
        "What is the interest rate on savings accounts?",
        "How much is the overdraft fee?",
        "What credit cards does TechBank offer?"
    ]
    
    while True:
        question = input("\n💬 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using Bank RAG System!")
            break
        
        if question.lower() == 'examples':
            print("\n📝 Example questions:")
            for i, q in enumerate(example_questions, 1):
                print(f"  {i}. {q}")
            continue
        
        if not question:
            continue
        
        # Process question
        print("\n🤔 Processing...")
        response = system.query(question)
        
        # Display response
        print("\n" + "-"*60)
        print(f"🤖 Answer:\n{response.answer}")
        print(f"\n📊 Confidence: {response.confidence:.0%}")
        print(f"⏱️  Response time: {response.response_time:.2f}s")
        
        if response.sources:
            print(f"\n📚 Sources:")
            for i, source in enumerate(response.sources, 1):
                print(f"  {i}. {source['source']}")
        
        print("-"*60)


def demo_batch():
    """Batch demo showing multiple queries"""
    print("\n" + "="*60)
    print("Bank RAG System - Batch Demo")
    print("="*60)
    
    # Initialize system
    system = BankRAGSystem()
    
    # Sample questions
    test_questions = [
        "What are the requirements to open a checking account?",
        "How do I use mobile deposit?",
        "Can I cancel a wire transfer?",
        "What is the minimum credit score for a loan?",
        "Where is the nearest ATM?"
    ]
    
    print(f"\nProcessing {len(test_questions)} questions...\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print('='*60)
        
        response = system.query(question)
        
        print(f"\nAnswer:\n{response.answer[:300]}...")
        print(f"\nConfidence: {response.confidence:.0%}")
        print(f"Response Time: {response.response_time:.2f}s")
        print(f"Sources: {len(response.sources)}")
        print(f"Refusal: {response.is_refusal}")


def demo_evaluation():
    """Run full evaluation against test dataset"""
    print("\n" + "="*60)
    print("Bank RAG System - Evaluation Demo")
    print("="*60)
    
    # Initialize system
    system = BankRAGSystem()
    
    # Run evaluation
    print("\nRunning evaluation on test queries...")
    test_queries = generate_test_queries()
    
    evaluator = RAGEvaluator(system.pipeline)
    metrics = evaluator.evaluate_dataset(test_queries)
    
    # Print results
    metrics.print_summary()
    
    # Check PRD requirements
    print("\n" + "="*60)
    print("PRD REQUIREMENTS CHECK")
    print("="*60)
    
    requirements = [
        ("Response Time", "< 3 seconds", f"{metrics.num_queries} queries averaged {system.pipeline.retriever}", "N/A (see individual queries)"),
        ("Retrieval Precision", "≥ 90%", f"{metrics.precision:.1%}", "✓ PASS" if metrics.precision >= 0.9 else "✗ NEEDS IMPROVEMENT"),
        ("Hit Rate", "≥ 85% (implied)", f"{metrics.hit_rate:.1%}", "✓ PASS" if metrics.hit_rate >= 0.85 else "✗ NEEDS IMPROVEMENT"),
    ]
    
    for req_name, target, actual, status in requirements:
        print(f"\n{req_name}:")
        print(f"  Target: {target}")
        print(f"  Actual: {actual}")
        if status != "N/A (see individual queries)":
            print(f"  Status: {status}")


def api_integration_example():
    """
    Example showing how full-stack team would integrate
    """
    print("\n" + "="*60)
    print("API Integration Example for Full-Stack Team")
    print("="*60)
    
    print("""
# Example FastAPI integration:

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import BankRAGSystem

app = FastAPI()
rag_system = BankRAGSystem()

class QuestionRequest(BaseModel):
    question: str
    include_sources: bool = True

class AnswerResponse(BaseModel):
    answer: str
    confidence: float
    sources: list
    is_refusal: bool
    response_time: float

@app.post("/api/v1/query", response_model=AnswerResponse)
async def query_endpoint(request: QuestionRequest):
    try:
        response = rag_system.query(
            request.question,
            include_sources=request.include_sources
        )
        return response.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Start server:
# uvicorn main:app --reload
""")
    
    print("\n✓ Integration template shown above")
    print("✓ RAGResponse objects are JSON-serializable")
    print("✓ All functionality available through BankRAGSystem class")


def main():
    """Main entry point"""
    import sys
    
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║           TechBank AI RAG-Based Agent System             ║
║                                                          ║
║  Intelligent Banking Assistant with Hybrid Retrieval    ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("Usage: python main.py [mode]")
        print("\nAvailable modes:")
        print("  interactive  - Interactive Q&A session")
        print("  batch        - Process multiple queries")
        print("  eval         - Run evaluation metrics")
        print("  api          - Show API integration example")
        print("\nDefault: interactive")
        print()
        mode = input("Select mode (or press Enter for interactive): ").strip().lower() or "interactive"
    
    # Run selected mode
    if mode == "interactive":
        demo_interactive()
    elif mode == "batch":
        demo_batch()
    elif mode in ["eval", "evaluation"]:
        demo_evaluation()
    elif mode == "api":
        api_integration_example()
    else:
        print(f"\n✗ Unknown mode: {mode}")
        print("Valid modes: interactive, batch, eval, api")
        sys.exit(1)


if __name__ == "__main__":
    main()
