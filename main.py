"""
Main Entry Point for Bank RAG System
Demonstrates setup and usage for full-stack team integration
"""

import logging
import json
import time
from typing import Optional
from document_loader import load_banking_documents
from chunking import chunk_banking_documents
from embeddings import create_embeddings_generator
from retriever import create_hybrid_retriever
from rag_pipeline import create_rag_pipeline, RAGResponse
from evaluation import ComprehensiveRAGEvaluator, generate_comprehensive_test_queries
from config import get_system_config
from monitoring import create_monitor

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
    
    def __init__(self, documents_path: str = "./documents", enable_monitoring: bool = True):
        """
        Initialize the complete RAG system
        
        Args:
            documents_path: Path to banking documents directory
            enable_monitoring: Whether to enable query monitoring
        """
        self.documents_path = documents_path
        self.pipeline = None
        self.enable_monitoring = enable_monitoring
        
        # Initialize monitoring if enabled
        self.monitor = create_monitor(log_dir="logs/queries") if self.enable_monitoring else None
        
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
        logger.info("\n3. Initializing embeddings...")
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
        # Track timing
        start_time = time.time()
        
        # Execute query
        response = self.pipeline.query(question, **kwargs)
        
        # Log to monitoring system if enabled
        if self.monitor:
            total_time_ms = (time.time() - start_time) * 1000
            
            # Extract retrieval scores
            retrieval_scores = []
            if hasattr(response, 'sources') and response.sources:
                for source in response.sources:
                    if isinstance(source, dict) and 'score' in source:
                        retrieval_scores.append(source['score'])
                    else:
                        retrieval_scores.append(0.8)
            
            # Get source documents for context
            retrieved_docs = []
            if hasattr(response, 'source_documents'):
                retrieved_docs = response.source_documents
            elif hasattr(response, 'sources'):
                # Convert sources to document-like objects if needed
                retrieved_docs = response.sources
            
            # Log the query
            self.monitor.log_query(
                query=question,
                answer=response.answer,
                retrieved_docs=retrieved_docs,
                retrieval_scores=retrieval_scores or [0.0],
                time_ms=total_time_ms,
                refused=response.is_refusal if hasattr(response, 'is_refusal') else False
            )
        
        return response
    
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
    print("Bank RAG System - Interactive Demo with Monitoring")
    print("="*60)
    print("\nInitializing system (this may take a minute)...")
    
    # Initialize system with monitoring enabled
    system = BankRAGSystem(enable_monitoring=True)
    
    print("\n✓ System ready! Ask questions about TechBank services.")
    print("📊 Monitoring enabled: All queries will be logged for analysis")
    print("Type 'quit' to exit, 'examples' for sample questions, 'stats' for session stats.\n")
    
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
            # Show session summary before exiting
            if system.monitor:
                system.monitor.print_session_summary()
            print("\n✨ Thank you for using Bank RAG System!")
            print("\n💡 Tip: Run 'python analyze_logs.py' to analyze all logged queries")
            break
        
        if question.lower() == 'examples':
            print("\n📝 Example questions:")
            for i, q in enumerate(example_questions, 1):
                print(f"  {i}. {q}")
            continue
        
        if question.lower() == 'stats':
            if system.monitor:
                system.monitor.print_session_summary()
            else:
                print("⚠️  Monitoring is disabled")
            continue
        
        if not question:
            continue
        
        # Process question
        print("\n🤔 Processing...")
        response = system.query(question)
        
        # Display response
        print("\n" + "-"*60)
        print(f"🤖 Answer:\n{response.answer}")
        
        print(f"\n📊 Metrics:")
        print(f"  Response time:  {response.response_time:.2f}s")
        print(f"  Refusal:        {'Yes ⚠️' if response.is_refusal else 'No ✓'}")
        print(f"  Retrieved docs: {len(response.sources)}")
        
        if response.sources:
            print(f"\n📚 Sources:")
            for i, source in enumerate(response.sources, 1):
                print(f"  {i}. {source['source']}")
        
        print("-"*60)


def demo_batch():
    """Batch demo showing multiple queries"""
    print("\n" + "="*60)
    print("Bank RAG System - Batch Demo with Monitoring")
    print("="*60)
    
    # Initialize system with monitoring
    system = BankRAGSystem(enable_monitoring=True)
    
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
    
    # Show session summary
    if system.monitor:
        system.monitor.print_session_summary()
    
    print("\n💡 Tip: Run 'python analyze_logs.py' to get detailed insights and optimization recommendations")


def demo_evaluation():
    """Run full evaluation against test dataset"""
    print("\n" + "="*60)
    print("Bank RAG System - Evaluation Demo")
    print("="*60)
    
    # Initialize system
    system = BankRAGSystem()
    
    # Run evaluation
    print("\nRunning evaluation on test queries...")
    test_queries = generate_comprehensive_test_queries()
    
    evaluator = ComprehensiveRAGEvaluator(system.pipeline)
    metrics = evaluator.evaluate_dataset(test_queries)
    
    # Print results
    metrics.print_summary()
    
    # Check PRD requirements
    print("\n" + "="*60)
    print("PRD REQUIREMENTS CHECK")
    print("="*60)
    
    requirements = [
        ("Response Time", "< 3 seconds", f"{metrics.num_queries} queries averaged {system.pipeline.retriever}", "N/A (see individual queries)"),
        ("Retrieval Precision", "≥ 90%", f"{metrics.retrieval.precision:.1%}", "✓ PASS" if metrics.retrieval.precision >= 0.9 else "✗ NEEDS IMPROVEMENT"),
        ("Hit Rate", "≥ 85% (implied)", f"{metrics.retrieval.hit_rate:.1%}", "✓ PASS" if metrics.retrieval.hit_rate >= 0.85 else "✗ NEEDS IMPROVEMENT"),
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
    
    print("\n💡 Quick Integration Guide:")
    print("  1. Import: from main import BankRAGSystem")
    print("  2. Initialize: rag = BankRAGSystem()")
    print("  3. Query: response = rag.query('Your question')")
    print("  4. Get result: response.answer, response.confidence, response.sources")
    print("\n📚 For full FastAPI example, see README.md")
    print("✓ RAGResponse.to_dict() is JSON-serializable")


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
        print("  interactive  - Interactive Q&A session with monitoring")
        print("  batch        - Process multiple queries and generate logs")
        print("  eval         - Run comprehensive evaluation metrics")
        print("  api          - Show API integration example")
        print("  analyze      - Analyze logs and get optimization suggestions")
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
    elif mode == "analyze":
        # Run log analysis
        import subprocess
        print("\n" + "="*60)
        print("Running Log Analysis...")
        print("="*60)
        subprocess.run(["python", "analyze_logs.py"])
    else:
        print(f"\n✗ Unknown mode: {mode}")
        print("Valid modes: interactive, batch, eval, api, analyze")
        sys.exit(1)


if __name__ == "__main__":
    main()
