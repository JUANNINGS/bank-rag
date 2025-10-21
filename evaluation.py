"""
Evaluation Module
Implements evaluation metrics for RAG system
Following patterns from RAG.ipynb: Hit Rate, MRR, nDCG, Precision
"""

import json
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from rag_pipeline import BankRAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Evaluation metrics results"""
    hit_rate: float  # Did we find the correct document?
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # Normalized Discounted Cumulative Gain
    precision: float  # Precision at k
    num_queries: int  # Number of queries evaluated
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def print_summary(self):
        """Print formatted summary"""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Number of queries: {self.num_queries}")
        print(f"Hit Rate:         {self.hit_rate:.2%} (found relevant doc)")
        print(f"MRR:              {self.mrr:.4f} (avg reciprocal rank)")
        print(f"nDCG:             {self.ndcg:.4f} (ranking quality)")
        print(f"Precision:        {self.precision:.2%} (relevant docs ratio)")
        print("="*60)


def compute_hit_rate(expected_ids: List[str], retrieved_ids: List[str]) -> float:
    """
    Hit Rate: Did we find any of the correct documents?
    
    Returns 1.0 if any retrieved document is correct, 0.0 otherwise
    
    Args:
        expected_ids: List of correct document IDs
        retrieved_ids: List of retrieved document IDs
        
    Returns:
        1.0 if hit, 0.0 if miss
    """
    is_hit = any(doc_id in expected_ids for doc_id in retrieved_ids)
    return 1.0 if is_hit else 0.0


def compute_mrr(expected_ids: List[str], retrieved_ids: List[str]) -> float:
    """
    MRR (Mean Reciprocal Rank): How highly ranked was the first correct document?
    
    Score = 1 / (position of first correct document)
    
    Examples:
    - Correct doc is 1st: MRR = 1/1 = 1.0
    - Correct doc is 2nd: MRR = 1/2 = 0.5
    - Correct doc is 3rd: MRR = 1/3 = 0.33
    - Correct doc not found: MRR = 0.0
    
    Args:
        expected_ids: List of correct document IDs
        retrieved_ids: List of retrieved document IDs
        
    Returns:
        Reciprocal rank score
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in expected_ids:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg(expected_ids: List[str], retrieved_ids: List[str]) -> float:
    """
    nDCG (Normalized Discounted Cumulative Gain): Quality-weighted ranking
    
    Considers all correct documents, not just the first
    Higher positions contribute more to the score
    
    Args:
        expected_ids: List of correct document IDs
        retrieved_ids: List of retrieved document IDs
        
    Returns:
        nDCG score between 0 and 1
    """
    if not retrieved_ids:
        return 0.0
    
    dcg = 0.0
    idcg = 0.0
    
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in expected_ids:
            dcg += 1.0 / (i + 1)
        idcg += 1.0 / (i + 1)
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_precision_at_k(expected_ids: List[str], retrieved_ids: List[str]) -> float:
    """
    Precision@k: What percentage of retrieved docs are relevant?
    
    Args:
        expected_ids: List of correct document IDs
        retrieved_ids: List of retrieved document IDs
        
    Returns:
        Precision score between 0 and 1
    """
    if not retrieved_ids:
        return 0.0
    
    relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in expected_ids)
    return relevant_retrieved / len(retrieved_ids)


class RAGEvaluator:
    """
    Evaluator for RAG system
    Tests retrieval quality using generated Q&A pairs
    """
    
    def __init__(self, pipeline: BankRAGPipeline):
        """
        Initialize evaluator
        
        Args:
            pipeline: RAG pipeline to evaluate
        """
        self.pipeline = pipeline
        logger.info("Initialized RAG Evaluator")
    
    def evaluate_query(
        self,
        query: str,
        expected_doc_ids: List[str],
        retrieved_docs
    ) -> Dict[str, float]:
        """
        Evaluate a single query
        
        Args:
            query: Test query
            expected_doc_ids: Ground truth document IDs
            retrieved_docs: Retrieved documents
            
        Returns:
            Dictionary of metrics
        """
        # Extract retrieved document IDs
        retrieved_ids = [
            doc.metadata.get('chunk_id', doc.metadata.get('source', f'doc_{i}'))
            for i, doc in enumerate(retrieved_docs)
        ]
        
        # Compute all metrics
        metrics = {
            'hit_rate': compute_hit_rate(expected_doc_ids, retrieved_ids),
            'mrr': compute_mrr(expected_doc_ids, retrieved_ids),
            'ndcg': compute_ndcg(expected_doc_ids, retrieved_ids),
            'precision': compute_precision_at_k(expected_doc_ids, retrieved_ids)
        }
        
        return metrics
    
    def evaluate_dataset(self, test_queries: List[Dict[str, any]]) -> EvaluationMetrics:
        """
        Evaluate entire test dataset
        
        Args:
            test_queries: List of test queries, each with:
                - 'query': question string
                - 'expected_sources': list of expected source names
                
        Returns:
            EvaluationMetrics with averaged results
        """
        logger.info(f"Evaluating {len(test_queries)} test queries...")
        
        all_metrics = {
            'hit_rate': [],
            'mrr': [],
            'ndcg': [],
            'precision': []
        }
        
        for i, test_item in enumerate(test_queries, 1):
            query = test_item['query']
            expected_sources = test_item['expected_sources']
            
            logger.debug(f"Query {i}/{len(test_queries)}: {query}")
            
            # Retrieve documents
            retrieved_docs = self.pipeline.retriever.invoke(query)
            
            # Evaluate
            metrics = self.evaluate_query(query, expected_sources, retrieved_docs)
            
            # Store metrics
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
        
        # Calculate averages
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        
        results = EvaluationMetrics(
            hit_rate=avg_metrics['hit_rate'],
            mrr=avg_metrics['mrr'],
            ndcg=avg_metrics['ndcg'],
            precision=avg_metrics['precision'],
            num_queries=len(test_queries)
        )
        
        logger.info("Evaluation complete")
        return results


def generate_test_queries() -> List[Dict[str, any]]:
    """
    Generate test Q&A pairs for banking domain
    
    Returns:
        List of test query dictionaries
    """
    test_queries = [
        {
            'query': 'What are the fees for wire transfers?',
            'expected_sources': ['wire_transfer_guide.txt']
        },
        {
            'query': 'How do I apply for a personal loan?',
            'expected_sources': ['loan_policy.txt']
        },
        {
            'query': 'What is the interest rate on savings accounts?',
            'expected_sources': ['account_types.txt']
        },
        {
            'query': 'How much is the overdraft fee?',
            'expected_sources': ['overdraft_protection.txt']
        },
        {
            'query': 'What credit cards does TechBank offer?',
            'expected_sources': ['credit_card_faq.txt']
        },
        {
            'query': 'Where can I find the nearest ATM?',
            'expected_sources': ['atm_locations.txt']
        },
        {
            'query': 'How do I use mobile deposit?',
            'expected_sources': ['mobile_banking_guide.txt']
        },
        {
            'query': 'What is the minimum credit score for a personal loan?',
            'expected_sources': ['loan_policy.txt']
        },
        {
            'query': 'Can I cancel a wire transfer after sending it?',
            'expected_sources': ['wire_transfer_guide.txt']
        },
        {
            'query': 'What are the requirements to open a checking account?',
            'expected_sources': ['account_types.txt']
        },
    ]
    
    return test_queries


def save_evaluation_results(metrics: EvaluationMetrics, filepath: str = "./tests/evaluation_results.json"):
    """
    Save evaluation results to JSON file
    
    Args:
        metrics: Evaluation metrics to save
        filepath: Path to save results
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    
    logger.info(f"Saved evaluation results to {filepath}")


def load_evaluation_results(filepath: str = "./tests/evaluation_results.json") -> EvaluationMetrics:
    """
    Load evaluation results from JSON file
    
    Args:
        filepath: Path to results file
        
    Returns:
        EvaluationMetrics object
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return EvaluationMetrics(**data)


# Example usage
if __name__ == "__main__":
    from document_loader import load_banking_documents
    from chunking import chunk_banking_documents
    from embeddings import create_embeddings_generator
    from retriever import create_hybrid_retriever
    from rag_pipeline import create_rag_pipeline
    
    print("="*60)
    print("Running RAG System Evaluation")
    print("="*60)
    
    try:
        # 1. Setup pipeline
        print("\n1. Setting up RAG pipeline...")
        documents = load_banking_documents("./documents")
        chunks = chunk_banking_documents(documents)
        embeddings_gen = create_embeddings_generator()
        embeddings_model = embeddings_gen.get_embeddings_model()
        retriever = create_hybrid_retriever(chunks, embeddings_model)
        pipeline = create_rag_pipeline(retriever)
        print("✓ Pipeline ready")
        
        # 2. Generate test queries
        print("\n2. Loading test queries...")
        test_queries = generate_test_queries()
        print(f"✓ Loaded {len(test_queries)} test queries")
        
        # 3. Run evaluation
        print("\n3. Running evaluation...")
        evaluator = RAGEvaluator(pipeline)
        metrics = evaluator.evaluate_dataset(test_queries)
        
        # 4. Print results
        metrics.print_summary()
        
        # 5. Save results
        save_evaluation_results(metrics)
        print("\n✓ Results saved to ./tests/evaluation_results.json")
        
        # 6. Check against PRD requirements
        print("\n" + "="*60)
        print("PRD REQUIREMENTS CHECK")
        print("="*60)
        print(f"Target: Retrieval Precision ≥ 90%")
        print(f"Actual: {metrics.precision:.2%}")
        print(f"Status: {'✓ PASS' if metrics.precision >= 0.9 else '✗ FAIL'}")
        print()
        print(f"Target: Hit Rate (implicit accuracy target)")
        print(f"Actual: {metrics.hit_rate:.2%}")
        print(f"Status: {'✓ GOOD' if metrics.hit_rate >= 0.85 else '⚠ NEEDS IMPROVEMENT'}")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
