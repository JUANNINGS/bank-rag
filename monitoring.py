"""
Simple Production Monitoring for RAG System
Records queries and key metrics with minimal overhead
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


class SimpleMonitor:
    """
    Lightweight query monitoring system
    
    Features:
    - Automatic query logging (JSONL format)
    - Daily log rotation
    - Session statistics
    """
    
    def __init__(self, log_dir: str = "logs/queries"):
        """
        Initialize monitor
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Session stats
        self.session_stats = {
            "total": 0,
            "total_time_ms": 0.0,
            "refused": 0,
            "total_score": 0.0
        }
    
    def _get_log_file(self) -> Path:
        """Get today's log file (daily rotation)"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"queries_{today}.jsonl"
    
    def log_query(self,
                  query: str,
                  answer: str,
                  retrieved_docs: List[Any],
                  retrieval_scores: List[float],
                  time_ms: float,
                  refused: bool = False) -> Dict:
        """
        Log a single query
        
        Args:
            query: User's question
            answer: System's answer
            retrieved_docs: Retrieved documents
            retrieval_scores: Relevance scores
            time_ms: Total response time (ms)
            refused: Whether system refused to answer
            
        Returns:
            Log entry dict
        """
        # Create log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer,
            "answer_length": len(answer),
            "num_docs": len(retrieved_docs),
            "avg_score": sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0,
            "time_ms": time_ms,
            "refused": refused,
            # Store context for Ragas evaluation
            "contexts": [
                doc.page_content if hasattr(doc, 'page_content') else str(doc)
                for doc in retrieved_docs
            ]
        }
        
        # Write to log file
        with open(self._get_log_file(), 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        # Update session stats
        self.session_stats["total"] += 1
        self.session_stats["total_time_ms"] += time_ms
        self.session_stats["total_score"] += log_entry["avg_score"]
        if refused:
            self.session_stats["refused"] += 1
        
        return log_entry
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        total = self.session_stats["total"]
        if total == 0:
            return self.session_stats
        
        return {
            "total_queries": total,
            "avg_time_ms": self.session_stats["total_time_ms"] / total,
            "avg_score": self.session_stats["total_score"] / total,
            "refused_count": self.session_stats["refused"],
            "refusal_rate": self.session_stats["refused"] / total
        }
    
    def print_session_summary(self):
        """Print session summary"""
        stats = self.get_session_stats()
        
        print("\n" + "="*60)
        print("📊 Session Summary")
        print("="*60)
        print(f"Total Queries:       {stats.get('total_queries', 0)}")
        print(f"Avg Response Time:   {stats.get('avg_time_ms', 0):.0f} ms")
        print(f"Avg Retrieval Score: {stats.get('avg_score', 0):.3f}")
        print(f"Refused:             {stats.get('refused_count', 0)} ({stats.get('refusal_rate', 0)*100:.1f}%)")
        print(f"Log File:            {self._get_log_file()}")
        print("="*60)
        print("\n💡 Tip: Run 'python analyze_logs.py' for detailed analysis")


# Convenience function
def create_monitor(log_dir: str = "logs/queries") -> SimpleMonitor:
    """Create a SimpleMonitor instance"""
    return SimpleMonitor(log_dir)
