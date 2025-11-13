"""
Log Analysis with Ragas Evaluation
Analyzes query logs and provides optimization recommendations using professional metrics
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta
import statistics

# Import Ragas for quality evaluation
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall
    )
    from datasets import Dataset
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("⚠️  Ragas not available. Install with: pip install ragas")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class LogAnalyzer:
    """
    Analyzes RAG query logs with Ragas metrics
    
    Provides:
    - Basic performance statistics
    - Ragas-based quality evaluation
    - Actionable optimization recommendations
    """
    
    def __init__(self, log_dir: str = "logs/queries"):
        """Initialize analyzer"""
        self.log_dir = Path(log_dir)
        self.logs = []
        self.load_logs()
    
    def load_logs(self, days: int = 7):
        """Load logs from last N days"""
        self.logs = []
        
        if not self.log_dir.exists():
            print(f"⚠️  Log directory not found: {self.log_dir}")
            return
        
        log_files = sorted(self.log_dir.glob("queries_*.jsonl"))
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for log_file in log_files:
            try:
                file_date_str = log_file.stem.replace("queries_", "")
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
                
                if file_date >= cutoff_date:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                self.logs.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            except ValueError:
                continue
        
        print(f"✓ Loaded {len(self.logs)} queries from last {days} days")
    
    def basic_stats(self) -> Dict[str, Any]:
        """Calculate basic statistics"""
        if not self.logs:
            return {}
        
        times = [log['time_ms'] for log in self.logs]
        scores = [log['avg_score'] for log in self.logs]
        
        return {
            "total_queries": len(self.logs),
            "avg_time_ms": statistics.mean(times),
            "p95_time_ms": statistics.quantiles(times, n=20)[18] if len(times) > 20 else max(times),
            "avg_retrieval_score": statistics.mean(scores),
            "refused_count": sum(1 for log in self.logs if log.get('refused', False)),
            "slow_queries": sum(1 for t in times if t > 3000),
            "poor_retrievals": sum(1 for s in scores if s < 0.6)
        }
    
    def evaluate_with_ragas(self, sample_size: int = 20) -> Dict[str, float]:
        """
        Evaluate query quality using Ragas metrics
        
        Args:
            sample_size: Number of queries to evaluate (for speed)
            
        Returns:
            Dict of Ragas metrics
        """
        if not RAGAS_AVAILABLE:
            return {}
        
        if not self.logs:
            return {}
        
        # Check if Azure OpenAI is configured
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        if not api_key or not endpoint:
            print("⚠️  Azure OpenAI not configured. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in .env")
            return {}
        
        # Sample queries (evaluate subset to save time)
        import random
        sampled_logs = random.sample(self.logs, min(sample_size, len(self.logs)))
        
        # Prepare data for Ragas
        data = {
            "question": [log['query'] for log in sampled_logs],
            "answer": [log['answer'] for log in sampled_logs],
            "contexts": [log.get('contexts', []) for log in sampled_logs]
        }
        
        dataset = Dataset.from_dict(data)
        
        print(f"\n🔍 Evaluating {len(sampled_logs)} queries with Ragas...")
        print("This may take 1-2 minutes...")
        
        try:
            # Configure Azure OpenAI for Ragas
            azure_model = AzureChatOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version="2024-02-15-preview",
                deployment_name=os.getenv("AZURE_GPT_DEPLOYMENT", "gpt-4"),
                model="gpt-4",
                temperature=0
            )
            
            azure_embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version="2024-02-15-preview",
                deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
            )
            
            # Run Ragas evaluation with Azure OpenAI
            results = evaluate(
                dataset,
                metrics=[
                    faithfulness,        # How faithful is answer to context?
                    answer_relevancy,    # How relevant is answer to question?
                ],
                llm=azure_model,
                embeddings=azure_embeddings
            )
            
            # Ragas returns lists of scores, calculate averages
            faithfulness_scores = results['faithfulness']
            relevancy_scores = results['answer_relevancy']
            
            return {
                "faithfulness": statistics.mean(faithfulness_scores) if isinstance(faithfulness_scores, list) else faithfulness_scores,
                "answer_relevancy": statistics.mean(relevancy_scores) if isinstance(relevancy_scores, list) else relevancy_scores,
                "sample_size": len(sampled_logs)
            }
        except Exception as e:
            print(f"⚠️  Ragas evaluation failed: {e}")
            print("    Continuing with basic analysis only...")
            return {}
    
    def identify_issues(self, stats: Dict, ragas_metrics: Dict) -> List[Dict]:
        """
        Generate prioritized recommendations
        
        Args:
            stats: Basic statistics
            ragas_metrics: Ragas evaluation results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # 1. Performance issues
        if stats.get('avg_time_ms', 0) > 3000:
            recommendations.append({
                "priority": "🔴 HIGH",
                "issue": f"Slow response: {stats['avg_time_ms']:.0f}ms avg (target: <3000ms)",
                "action": [
                    "Reduce k parameter (fewer docs to retrieve)",
                    "Optimize vector index",
                    "Consider caching common queries"
                ]
            })
        
        # 2. Retrieval quality (traditional metric)
        poor_rate = stats.get('poor_retrievals', 0) / stats.get('total_queries', 1)
        if poor_rate > 0.2:
            recommendations.append({
                "priority": "🔴 HIGH",
                "issue": f"Poor retrieval quality: {poor_rate*100:.1f}% queries <0.6 score",
                "action": [
                    "Adjust chunk_size: try 500-800 instead of 1000",
                    "Increase chunk_overlap: try 200 instead of 100",
                    "Add more diverse documents to knowledge base"
                ]
            })
        
        # 3. Faithfulness (Ragas metric)
        if ragas_metrics.get('faithfulness'):
            if ragas_metrics['faithfulness'] < 0.7:
                recommendations.append({
                    "priority": "🔴 HIGH",
                    "issue": f"Low faithfulness: {ragas_metrics['faithfulness']:.2f} (target: >0.7)",
                    "impact": "System is hallucinating or adding information not in documents",
                    "action": [
                        "Strengthen prompt: 'Answer ONLY based on provided context'",
                        "Add explicit instruction: 'Do not add external knowledge'",
                        "Lower temperature: try 0.3 instead of 0.7"
                    ]
                })
        
        # 4. Answer relevancy (Ragas metric)
        if ragas_metrics.get('answer_relevancy'):
            if ragas_metrics['answer_relevancy'] < 0.7:
                recommendations.append({
                    "priority": "🟡 MEDIUM",
                    "issue": f"Low answer relevancy: {ragas_metrics['answer_relevancy']:.2f} (target: >0.7)",
                    "impact": "Answers are not directly addressing user questions",
                    "action": [
                        "Improve prompt: Add 'Answer the question directly and concisely'",
                        "Review retrieved documents: Are they actually relevant?",
                        "Consider query rewriting/expansion"
                    ]
                })
        
        # 5. High refusal rate
        refusal_rate = stats.get('refused_count', 0) / stats.get('total_queries', 1)
        if refusal_rate > 0.15:
            recommendations.append({
                "priority": "🟡 MEDIUM",
                "issue": f"High refusal rate: {refusal_rate*100:.1f}%",
                "action": [
                    "Review refused queries - are they legitimate?",
                    "If yes: lower confidence threshold",
                    "If no: add FAQ for common out-of-scope questions"
                ]
            })
        
        return recommendations
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        
        print("\n" + "="*70)
        print("📊 RAG SYSTEM ANALYSIS REPORT")
        print("="*70)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        if not self.logs:
            print("\n⚠️  No logs found!")
            print("\nTo generate logs:")
            print("  1. Run: python main.py interactive")
            print("  2. Ask some questions")
            print("  3. Run this analyzer again")
            return
        
        # Basic statistics
        print("\n📈 1. BASIC STATISTICS")
        print("-" * 70)
        stats = self.basic_stats()
        print(f"Total Queries:          {stats['total_queries']}")
        print(f"Avg Response Time:      {stats['avg_time_ms']:.0f} ms")
        print(f"95th Percentile Time:   {stats['p95_time_ms']:.0f} ms")
        print(f"Avg Retrieval Score:    {stats['avg_retrieval_score']:.3f}")
        print(f"Slow Queries (>3s):     {stats['slow_queries']} ({stats['slow_queries']/stats['total_queries']*100:.1f}%)")
        print(f"Poor Retrievals (<0.6): {stats['poor_retrievals']} ({stats['poor_retrievals']/stats['total_queries']*100:.1f}%)")
        print(f"Refused Queries:        {stats['refused_count']} ({stats['refused_count']/stats['total_queries']*100:.1f}%)")
        
        # Ragas evaluation
        print("\n🎯 2. RAGAS QUALITY METRICS")
        print("-" * 70)
        
        if RAGAS_AVAILABLE:
            ragas_metrics = self.evaluate_with_ragas(sample_size=20)
            
            if ragas_metrics:
                print(f"Sample Size:            {ragas_metrics.get('sample_size', 0)} queries")
                print(f"Faithfulness:           {ragas_metrics.get('faithfulness', 0):.3f} (target: >0.7)")
                print(f"Answer Relevancy:       {ragas_metrics.get('answer_relevancy', 0):.3f} (target: >0.7)")
                
                print("\n📖 Metric Explanation:")
                print("  • Faithfulness: Does answer stay true to retrieved documents?")
                print("  • Answer Relevancy: Does answer directly address the question?")
            else:
                print("⚠️  Ragas evaluation failed")
                ragas_metrics = {}
        else:
            print("⚠️  Ragas not installed. Install with: pip install ragas")
            ragas_metrics = {}
        
        # Recommendations
        print("\n💡 3. OPTIMIZATION RECOMMENDATIONS")
        print("="*70)
        
        recommendations = self.identify_issues(stats, ragas_metrics)
        
        if not recommendations:
            print("✅ No major issues detected! System is performing well.")
        else:
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{rec['priority']} Recommendation #{i}")
                print(f"Issue:  {rec['issue']}")
                if 'impact' in rec:
                    print(f"Impact: {rec['impact']}")
                print("Actions:")
                for action in rec['action']:
                    print(f"  • {action}")
        
        # Sample problematic queries
        print("\n🔍 4. SAMPLE ISSUES")
        print("-" * 70)
        
        poor_queries = [log for log in self.logs if log['avg_score'] < 0.6]
        if poor_queries:
            print(f"\nPoor Retrieval Samples (showing up to 5):")
            for log in poor_queries[:5]:
                print(f"  Q: {log['query'][:80]}")
                print(f"     Score: {log['avg_score']:.3f}, Docs: {log['num_docs']}")
        
        slow_queries = [log for log in self.logs if log['time_ms'] > 3000]
        if slow_queries:
            print(f"\nSlow Query Samples (showing up to 3):")
            for log in slow_queries[:3]:
                print(f"  Q: {log['query'][:80]}")
                print(f"     Time: {log['time_ms']:.0f}ms")
        
        print("\n" + "="*70)
        print("✨ Analysis complete!")
        print("\n📋 Next Steps:")
        print("  1. Review recommendations above (prioritize 🔴 HIGH)")
        print("  2. Implement changes one at a time")
        print("  3. Re-run analysis after changes to measure impact")
        print("="*70)


def main():
    """Main entry point"""
    analyzer = LogAnalyzer(log_dir="logs/queries")
    analyzer.generate_report()


if __name__ == "__main__":
    main()
