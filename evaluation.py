"""
Bank RAG Evaluation Module - Streamlined Version
================================================================================
Three-tier evaluation system:
1. Retrieval Quality: Are the retrieved documents correct?
2. Generation Quality: Is the answer hallucination-free? Does it answer the question?
3. End-to-End: How does the answer compare to the reference answer?
================================================================================
"""

import json
import logging
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
import numpy as np

# LangChain and Ragas imports
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness,           # Faithfulness: Detect hallucinations
    answer_relevancy,       # Answer relevancy
    context_recall,         # Context recall
    answer_correctness,     # Answer correctness
    answer_similarity,      # Answer similarity
)
from ragas.dataset_schema import SingleTurnSample

from rag_pipeline import BankRAGPipeline
from config import get_model_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 📊 Part 1: Data Structure Definitions
# ============================================================================
# Why needed?: Organize evaluation results for storage and display

@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics: Are the retrieved documents correct?"""
    hit_rate: float          # Hit rate: Was at least one relevant document found?
    mrr: float               # Mean Reciprocal Rank: What position is the relevant document?
    ndcg: float              # Normalized Discounted Cumulative Gain: Overall ranking quality
    precision: float         # Precision: What proportion of retrieved documents are relevant?
    context_recall: Optional[float] = None  # Recall: Are there any missed relevant documents?


@dataclass
class GenerationMetrics:
    """Generation quality metrics: Is the answer good?"""
    faithfulness: Optional[float] = None        # Faithfulness: Any hallucinations? (Most important!)
    answer_relevancy: Optional[float] = None    # Relevancy: Does it answer the question?
    answer_correctness: Optional[float] = None  # Correctness: Comparison with reference answer
    answer_similarity: Optional[float] = None   # Similarity: Semantic similarity level


@dataclass
class ComprehensiveMetrics:
    """Complete evaluation results"""
    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    num_queries: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            'retrieval': asdict(self.retrieval),
            'generation': asdict(self.generation),
            'num_queries': self.num_queries,
            'timestamp': self.timestamp
        }
    
    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*70)
        print("📊 RAG System Evaluation Results")
        print("="*70)
        print(f"Number of test queries: {self.num_queries}")
        
        print("\n🔍 Retrieval Quality")
        print(f"  Precision:      {self.retrieval.precision:.1%}  (Target: ≥90%)")
        print(f"  Hit Rate:       {self.retrieval.hit_rate:.1%}")
        print(f"  MRR:            {self.retrieval.mrr:.3f}")
        if self.retrieval.context_recall:
            print(f"  Context Recall: {self.retrieval.context_recall:.1%}")
        
        print("\n✨ Generation Quality")
        if self.generation.faithfulness:
            status = "✅" if self.generation.faithfulness >= 0.95 else "⚠️"
            print(f"  Faithfulness:   {self.generation.faithfulness:.1%}  {status} (Target: ≥95%)")
        if self.generation.answer_relevancy:
            status = "✅" if self.generation.answer_relevancy >= 0.85 else "⚠️"
            print(f"  Relevancy:      {self.generation.answer_relevancy:.1%}  {status} (Target: ≥85%)")
        if self.generation.answer_correctness:
            print(f"  Correctness:    {self.generation.answer_correctness:.1%}")
        print("="*70 + "\n")


# ============================================================================
# 📐 Part 2: Traditional Retrieval Metrics Calculation
# ============================================================================
# Why needed?: These are basic metrics that don't require LLM, fast and low-cost

def compute_hit_rate(expected_ids: List[str], retrieved_ids: List[str]) -> float:
    """
    Hit Rate: Was at least one relevant document found?
    Returns: 1.0 (found) or 0.0 (not found)
    """
    return 1.0 if any(doc_id in expected_ids for doc_id in retrieved_ids) else 0.0


def compute_mrr(expected_ids: List[str], retrieved_ids: List[str]) -> float:
    """
    Mean Reciprocal Rank (MRR): What position is the first relevant document?
    Example: Position 1 → 1.0, Position 2 → 0.5, Position 3 → 0.33
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in expected_ids:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg(expected_ids: List[str], retrieved_ids: List[str]) -> float:
    """
    Normalized Discounted Cumulative Gain (nDCG): What's the overall ranking quality?
    Considers all relevant documents, with higher positions contributing more
    """
    if not retrieved_ids:
        return 0.0
    
    dcg = sum(1.0 / (i + 1) for i, doc_id in enumerate(retrieved_ids) if doc_id in expected_ids)
    idcg = sum(1.0 / (i + 1) for i in range(len(retrieved_ids)))
    return dcg / idcg if idcg > 0 else 0.0


def compute_precision_at_k(expected_ids: List[str], retrieved_ids: List[str]) -> float:
    """
    Precision: What proportion of retrieved documents are relevant?
    Example: Retrieved 5 documents, 3 relevant → 60%
    """
    if not retrieved_ids:
        return 0.0
    relevant = sum(1 for doc_id in retrieved_ids if doc_id in expected_ids)
    return relevant / len(retrieved_ids)


# ============================================================================
# 🤖 Part 3: Ragas Evaluator (Using LLM for Advanced Evaluation)
# ============================================================================
# Why needed?: Traditional metrics only evaluate retrieval, can't evaluate generation quality (e.g., hallucination detection)

class RagasEvaluator:
    """
    Ragas Evaluator: Using LLM to evaluate generation quality
    
    How it works:
    1. Use GPT as a "judge"
    2. Provide it with question, answer, and retrieved documents
    3. Ask it to judge: Is the answer faithful? Relevant? Correct?
    """
    
    def __init__(self):
        """Initialize LLM and embeddings using Open Source models"""
        config = get_model_config()
        
        # Initialize Ollama LLM
        llm = ChatOllama(
            base_url=config.ollama_base_url,
            model=config.llm_model,
            temperature=0.0  # Use 0 temperature for evaluation to ensure determinism
        )
        
        # Initialize HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Wrap in Ragas format
        self.ragas_llm = LangchainLLMWrapper(llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
        
        # Configure LLM for each metric
        faithfulness.llm = self.ragas_llm
        answer_relevancy.llm = self.ragas_llm
        answer_relevancy.embeddings = self.ragas_embeddings
        context_recall.llm = self.ragas_llm
        answer_correctness.llm = self.ragas_llm
        answer_similarity.llm = self.ragas_llm
        answer_similarity.embeddings = self.ragas_embeddings
        answer_correctness.answer_similarity = answer_similarity
        
        logger.info("✓ Ragas evaluator initialized")
    
    async def evaluate_faithfulness(self, query: str, response: str, contexts: List[str]) -> float:
        """
        Evaluate faithfulness: Is the answer faithful to the retrieved documents?
        
        How it works:
        1. Extract statements from the answer
        2. Check if each statement can be supported by the contexts
        3. Return: (number of supported statements) / (total statements)
        
        Example:
        Context: "Loan interest rate is 6.99%-15.99%"
        Answer: "Loan interest rate is 5%"  → 0% (Hallucination!)
        Answer: "Loan interest rate is 6.99%-15.99%" → 100% (Faithful)
        """
        try:
            sample = SingleTurnSample(
                user_input=query,
                response=response,
                retrieved_contexts=contexts
            )
            score = await faithfulness.single_turn_ascore(sample)
            return float(score)
        except Exception as e:
            logger.warning(f"Faithfulness evaluation failed: {e}")
            return None
    
    async def evaluate_answer_relevancy(self, query: str, response: str) -> float:
        """
        Evaluate answer relevancy: Does the answer truly address the question?
        
        How it works:
        1. Use LLM to generate questions from the answer (reverse engineering)
        2. Compare similarity between generated questions and original question
        3. If answer is relevant, generated questions should be similar to original
        
        Example:
        Question: "How to cancel a transfer?"
        Answer: "You can view transfer history" → Low score (Doesn't answer)
        Answer: "Transfers cannot be cancelled" → High score (Direct answer)
        """
        try:
            sample = SingleTurnSample(user_input=query, response=response)
            score = await answer_relevancy.single_turn_ascore(sample)
            return float(score)
        except Exception as e:
            logger.warning(f"Answer Relevancy evaluation failed: {e}")
            return None
    
    async def evaluate_context_recall(self, query: str, reference: str, contexts: List[str]) -> float:
        """
        Evaluate context recall: Do the retrieved documents contain all information from the reference answer?
        
        How it works:
        1. Extract key information points from the reference answer
        2. Check if each information point can be found in the retrieved documents
        3. Return: (found information points) / (total information points)
        """
        try:
            sample = SingleTurnSample(
                user_input=query,
                reference=reference,
                retrieved_contexts=contexts
            )
            score = await context_recall.single_turn_ascore(sample)
            return float(score)
        except Exception as e:
            logger.warning(f"Context Recall evaluation failed: {e}")
            return None
    
    async def evaluate_answer_correctness(self, query: str, response: str, reference: str) -> Tuple[float, float]:
        """
        Evaluate answer correctness: How does the answer compare to the reference answer?
        
        Returns two scores:
        1. Correctness = 0.75 * factual correctness + 0.25 * semantic similarity
        2. Similarity = pure semantic similarity
        """
        try:
            sample = SingleTurnSample(
                user_input=query,
                response=response,
                reference=reference
            )
            correctness_score = await answer_correctness.single_turn_ascore(sample)
            similarity_score = await answer_similarity.single_turn_ascore(sample)
            return float(correctness_score), float(similarity_score)
        except Exception as e:
            logger.warning(f"Answer Correctness evaluation failed: {e}")
            return None, None


# ============================================================================
# 🎯 Part 4: Comprehensive Evaluator (Core)
# ============================================================================
# Why needed?: Integrate all evaluation functions, provide unified interface

class ComprehensiveRAGEvaluator:
    """
    Comprehensive RAG Evaluator: Integrating traditional and Ragas metrics
    
    Evaluation process:
    1. For each test question
    2. Get RAG system's answer
    3. Calculate traditional retrieval metrics (fast)
    4. Calculate Ragas generation metrics (slow, requires LLM)
    5. Aggregate results
    """
    
    def __init__(self, pipeline: BankRAGPipeline, use_ragas: bool = True):
        self.pipeline = pipeline
        self.use_ragas = use_ragas
        
        if use_ragas:
            try:
                self.ragas_evaluator = RagasEvaluator()
            except Exception as e:
                logger.warning(f"Ragas initialization failed: {e}. Using traditional metrics only")
                self.use_ragas = False
        
        logger.info("✓ Comprehensive evaluator initialized")
    
    async def evaluate_query_async(
        self,
        query: str,
        expected_sources: List[str],
        reference_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate a single query (async version for Ragas)"""
        
        # 1. Get RAG system's answer
        response = self.pipeline.query(query)
        
        # 2. Get retrieved documents
        retrieved_docs = self.pipeline.retriever.invoke(query)
        retrieved_ids = [
            doc.metadata.get('source', f'doc_{i}').split('/')[-1]
            for i, doc in enumerate(retrieved_docs)
        ]
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]
        
        # 3. Calculate traditional retrieval metrics (fast, no LLM needed)
        retrieval_metrics = {
            'hit_rate': compute_hit_rate(expected_sources, retrieved_ids),
            'mrr': compute_mrr(expected_sources, retrieved_ids),
            'ndcg': compute_ndcg(expected_sources, retrieved_ids),
            'precision': compute_precision_at_k(expected_sources, retrieved_ids)
        }
        
        # 4. Calculate Ragas metrics (slow, requires LLM calls)
        generation_metrics = {}
        
        if self.use_ragas and self.ragas_evaluator:
            # Faithfulness: Detect hallucinations (most important!)
            faith_score = await self.ragas_evaluator.evaluate_faithfulness(
                query, response.answer, retrieved_contexts
            )
            generation_metrics['faithfulness'] = faith_score
            
            # Answer Relevancy: Does it answer the question?
            rel_score = await self.ragas_evaluator.evaluate_answer_relevancy(
                query, response.answer
            )
            generation_metrics['answer_relevancy'] = rel_score
            
            # If reference answer is provided, do more evaluation
            if reference_answer:
                # Context Recall: Retrieval completeness
                recall_score = await self.ragas_evaluator.evaluate_context_recall(
                    query, reference_answer, retrieved_contexts
                )
                retrieval_metrics['context_recall'] = recall_score
                
                # Answer Correctness: Answer correctness
                correct_score, sim_score = await self.ragas_evaluator.evaluate_answer_correctness(
                    query, response.answer, reference_answer
                )
                generation_metrics['answer_correctness'] = correct_score
                generation_metrics['answer_similarity'] = sim_score
        
        return {
            'query': query,
            'response': response.answer,
            'retrieval': retrieval_metrics,
            'generation': generation_metrics,
            'confidence': response.confidence,
            'sources': response.sources
        }
    
    async def evaluate_dataset_async(
        self,
        test_queries: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> ComprehensiveMetrics:
        """Evaluate entire test set (async)"""
        logger.info(f"🚀 Starting evaluation of {len(test_queries)} queries...")
        
        # Store all metrics
        all_retrieval = {
            'hit_rate': [], 'mrr': [], 'ndcg': [], 
            'precision': [], 'context_recall': []
        }
        all_generation = {
            'faithfulness': [], 'answer_relevancy': [],
            'answer_correctness': [], 'answer_similarity': []
        }
        
        # Evaluate each query
        for i, test_item in enumerate(test_queries, 1):
            if show_progress:
                print(f"  [{i}/{len(test_queries)}] {test_item['query'][:50]}...")
            
            result = await self.evaluate_query_async(
                query=test_item['query'],
                expected_sources=test_item['expected_sources'],
                reference_answer=test_item.get('reference_answer')
            )
            
            # Collect metrics
            for key in all_retrieval:
                if key in result['retrieval'] and result['retrieval'][key] is not None:
                    all_retrieval[key].append(result['retrieval'][key])
            
            for key in all_generation:
                if key in result['generation'] and result['generation'][key] is not None:
                    all_generation[key].append(result['generation'][key])
        
        # Calculate averages
        retrieval_metrics = RetrievalMetrics(
            hit_rate=np.mean(all_retrieval['hit_rate']) if all_retrieval['hit_rate'] else 0.0,
            mrr=np.mean(all_retrieval['mrr']) if all_retrieval['mrr'] else 0.0,
            ndcg=np.mean(all_retrieval['ndcg']) if all_retrieval['ndcg'] else 0.0,
            precision=np.mean(all_retrieval['precision']) if all_retrieval['precision'] else 0.0,
            context_recall=np.mean(all_retrieval['context_recall']) if all_retrieval['context_recall'] else None
        )
        
        generation_metrics = GenerationMetrics(
            faithfulness=np.mean(all_generation['faithfulness']) if all_generation['faithfulness'] else None,
            answer_relevancy=np.mean(all_generation['answer_relevancy']) if all_generation['answer_relevancy'] else None,
            answer_correctness=np.mean(all_generation['answer_correctness']) if all_generation['answer_correctness'] else None,
            answer_similarity=np.mean(all_generation['answer_similarity']) if all_generation['answer_similarity'] else None
        )
        
        results = ComprehensiveMetrics(
            retrieval=retrieval_metrics,
            generation=generation_metrics,
            num_queries=len(test_queries)
        )
        
        logger.info("✅ Evaluation completed!")
        return results

    def evaluate_dataset(self, test_queries: List[Dict[str, Any]], show_progress: bool = True) -> ComprehensiveMetrics:
        """Evaluate entire test set (sync wrapper)"""
        return asyncio.run(self.evaluate_dataset_async(test_queries, show_progress))


# ============================================================================
# 📝 Part 5: Test Dataset (Streamlined to 10 representative questions)
# ============================================================================
# Why streamlined?: 30 questions were too long (200+ lines), 10 are sufficient to cover main scenarios

def generate_comprehensive_test_queries() -> List[Dict[str, Any]]:
    """
    Generate test dataset: 10 representative questions covering 5 business categories
    
    Each test includes:
    - query: User question
    - expected_sources: Expected document sources
    - reference_answer: Reference answer (for correctness evaluation)
    - category: Business category
    """
    return [
        # 1. Personal Loans
        {
            'query': 'How do I apply for a personal loan?',
            'expected_sources': ['loan_policy.txt'],
            'reference_answer': 'To apply for a personal loan at Melbourne First Bank, you can apply online through internet banking, visit a branch, or call 1300-FIRST-BANK.',
            'category': 'loan'
        },
        {
            'query': 'What is the minimum credit score for a personal loan?',
            'expected_sources': ['loan_policy.txt'],
            'reference_answer': 'Melbourne First Bank typically requires a minimum credit score of 650 for personal loan approval.',
            'category': 'loan'
        },
        
        # 2. Credit Cards
        {
            'query': 'What credit cards does Melbourne First Bank offer?',
            'expected_sources': ['credit_card_faq.txt'],
            'reference_answer': 'Melbourne First Bank offers three main credit cards: Classic Card, Gold Rewards Card, and Platinum Card.',
            'category': 'credit_card'
        },
        {
            'query': 'What is the interest-free period on credit cards?',
            'expected_sources': ['credit_card_faq.txt'],
            'reference_answer': 'Melbourne First Bank credit cards offer up to 55 days interest-free on purchases when you pay your closing balance in full by the due date.',
            'category': 'credit_card'
        },
        
        # 3. Accounts
        {
            'query': 'What are the requirements to open a checking account?',
            'expected_sources': ['account_types.txt'],
            'reference_answer': 'To open a checking account, you need to be at least 18 years old, provide valid photo ID, proof of address, and an initial deposit of $100.',
            'category': 'account'
        },
        {
            'query': 'What is the interest rate on savings accounts?',
            'expected_sources': ['account_types.txt'],
            'reference_answer': 'Melbourne First Bank offers a standard savings account with a base rate of 0.50% p.a., and up to 4.25% p.a. bonus interest.',
            'category': 'account'
        },
        
        # 4. International Transfers
        {
            'query': 'What are the fees for wire transfers?',
            'expected_sources': ['wire_transfer_guide.txt'],
            'reference_answer': 'Melbourne First Bank charges $20 for domestic wire transfers and $30 for international wire transfers.',
            'category': 'transfer'
        },
        {
            'query': 'Can I cancel a wire transfer after sending it?',
            'expected_sources': ['wire_transfer_guide.txt'],
            'reference_answer': 'Domestic wire transfers cannot be cancelled once sent. International wire transfers may be cancelled within 1 hour if they haven\'t been processed.',
            'category': 'transfer'
        },
        
        # 5. ATM/Mobile Banking
        {
            'query': 'Where can I find the nearest ATM?',
            'expected_sources': ['atm_locations.txt'],
            'reference_answer': 'You can find the nearest ATM using the branch locator on our website or mobile app. We have ATMs throughout Victoria.',
            'category': 'atm'
        },
        {
            'query': 'How do I use mobile deposit?',
            'expected_sources': ['mobile_banking_guide.txt'],
            'reference_answer': 'To use mobile deposit, log into the app, select "Deposit Cheque", take photos of the cheque, enter the amount, and submit.',
            'category': 'mobile'
        },
    ]


# ============================================================================
# 💾 Part 6: Report Saving and Generation
# ============================================================================

def save_comprehensive_report(metrics: ComprehensiveMetrics, filepath: str = "./tests/evaluation_report.json"):
    """Save JSON report"""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    logger.info(f"✓ JSON report saved to: {filepath}")


def generate_html_report(metrics: ComprehensiveMetrics, filepath: str = "./tests/evaluation_report.html"):
    """Generate HTML report (simplified version)"""
    
    # Simplified HTML template
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Bank RAG Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        .metric {{ display: inline-block; width: 200px; margin: 15px; padding: 20px; background: #ecf0f1; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 32px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
        .pass {{ color: #27ae60; }}
        .fail {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏦 Bank RAG Evaluation Report</h1>
        <p>Evaluation time: {metrics.timestamp}</p>
        <p>Test queries: <strong>{metrics.num_queries}</strong></p>
        
        <h2>🔍 Retrieval Quality</h2>
        <div class="metric">
            <div class="metric-value">{metrics.retrieval.precision:.0%}</div>
            <div class="metric-label">Precision</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics.retrieval.hit_rate:.0%}</div>
            <div class="metric-label">Hit Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics.retrieval.mrr:.2f}</div>
            <div class="metric-label">MRR</div>
        </div>
        
        <h2>✨ Generation Quality</h2>
        {'<div class="metric"><div class="metric-value ' + ('pass' if metrics.generation.faithfulness >= 0.95 else 'fail') + f'">{metrics.generation.faithfulness:.0%}</div><div class="metric-label">Faithfulness (≥95%)</div></div>' if metrics.generation.faithfulness else ''}
        {'<div class="metric"><div class="metric-value ' + ('pass' if metrics.generation.answer_relevancy >= 0.85 else 'fail') + f'">{metrics.generation.answer_relevancy:.0%}</div><div class="metric-label">Relevancy (≥85%)</div></div>' if metrics.generation.answer_relevancy else ''}
        
        <h2>🎯 PRD Requirements Check</h2>
        <p>Precision ≥ 90%: <strong class="{'pass' if metrics.retrieval.precision >= 0.9 else 'fail'}">
            {metrics.retrieval.precision:.1%} {'✅ PASS' if metrics.retrieval.precision >= 0.9 else '❌ FAIL'}
        </strong></p>
        {'<p>Faithfulness ≥ 95%: <strong class="' + ('pass' if metrics.generation.faithfulness >= 0.95 else 'fail') + f'">{metrics.generation.faithfulness:.1%} {("✅ PASS" if metrics.generation.faithfulness >= 0.95 else "❌ FAIL")}</strong></p>' if metrics.generation.faithfulness else ''}
    </div>
</body>
</html>"""
    
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"✓ HTML report saved to: {filepath}")


# ============================================================================
# 🚀 Main Program: Run Evaluation
# ============================================================================

if __name__ == "__main__":
    from document_loader import load_banking_documents
    from chunking import chunk_banking_documents
    from embeddings import create_embeddings_generator
    from retriever import create_hybrid_retriever
    from rag_pipeline import create_rag_pipeline
    
    print("\n" + "="*70)
    print("🚀 Bank RAG System Evaluation")
    print("="*70)
    
    try:
        # 1. Initialize RAG system
        print("\n[1/5] Initializing RAG system...")
        documents = load_banking_documents("./documents")
        chunks = chunk_banking_documents(documents)
        embeddings_gen = create_embeddings_generator()
        embeddings_model = embeddings_gen.get_embeddings_model()
        retriever = create_hybrid_retriever(chunks, embeddings_model)
        pipeline = create_rag_pipeline(retriever)
        print("      ✓ RAG system ready")
        
        # 2. Load test data
        print("\n[2/5] Loading test data...")
        test_queries = generate_comprehensive_test_queries()
        print(f"      ✓ Loaded {len(test_queries)} test queries")
        
        # 3. Run evaluation
        print("\n[3/5] Running evaluation (this will take 5-10 minutes)...")
        evaluator = ComprehensiveRAGEvaluator(pipeline, use_ragas=True)
        metrics = evaluator.evaluate_dataset(test_queries, show_progress=True)
        
        # 4. Display results
        print("\n[4/5] Evaluation completed!")
        metrics.print_summary()
        
        # 5. Save reports
        print("\n[5/5] Generating reports...")
        save_comprehensive_report(metrics)
        generate_html_report(metrics)
        print("      ✓ JSON report: ./tests/evaluation_report.json")
        print("      ✓ HTML report: ./tests/evaluation_report.html")
        
        print("\n" + "="*70)
        print("✅ Evaluation completed! Open the HTML report in your browser for details")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
