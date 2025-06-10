from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any
from shared.logger_config import EnhancedLogger

@dataclass
class ChatbotMetrics:
    timestamp: datetime
    phase: str  # collection/qa
    duration_ms: float
    status: str
    details: Dict[str, Any]

class ChatbotMonitoring:
    def __init__(self) -> None:
        """Initialize monitoring metrics"""
        self.metrics = {
            "llm_calls": {
                "success": 0,
                "failed": 0,
                "total_time_ms": 0,
                "average_time_ms": 0
            },
            "rag_queries": {
                "total": 0,
                "no_matches": 0,
                "average_similarity": 0.0
            },
            "conversation": {
                "collection_phase": {"success": 0, "failed": 0},
                "qa_phase": {"success": 0, "failed": 0},
                "language_stats": {"he": 0, "en": 0}
            }
        }
        self.logger = EnhancedLogger()

    def log_llm_call(self, duration_ms: float, success: bool, **details) -> None:
        """Log LLM API call performance"""
        status = "success" if success else "failed"
        self.metrics["llm_calls"][status] += 1
        self.metrics["llm_calls"]["total_time_ms"] += duration_ms
        
        # Update average response time
        total_calls = self.metrics["llm_calls"]["success"] + self.metrics["llm_calls"]["failed"]
        self.metrics["llm_calls"]["average_time_ms"] = (
            self.metrics["llm_calls"]["total_time_ms"] / total_calls
        )
        
        self.logger.info("LLM Call Metrics",
            duration_ms=duration_ms,
            status=status,
            average_time=self.metrics["llm_calls"]["average_time_ms"],
            **details
        )

    def log_rag_query(self, similarity_score: float, found_match: bool, **details) -> None:
        """Log RAG query effectiveness"""
        self.metrics["rag_queries"]["total"] += 1
        if not found_match:
            self.metrics["rag_queries"]["no_matches"] += 1
        
        # Update average similarity score
        prev_avg = self.metrics["rag_queries"]["average_similarity"]
        n = self.metrics["rag_queries"]["total"]
        self.metrics["rag_queries"]["average_similarity"] = (
            (prev_avg * (n - 1) + similarity_score) / n
        )
        
        self.logger.info("RAG Query Metrics",
            similarity=similarity_score,
            found_match=found_match,
            no_match_rate=self.metrics["rag_queries"]["no_matches"] / n,
            **details
        )

    def log_conversation(self, phase: str, success: bool, language: str) -> None:
        """Log conversation flow metrics"""
        status = "success" if success else "failed"
        self.metrics["conversation"][f"{phase}_phase"][status] += 1
        self.metrics["conversation"]["language_stats"][language] += 1
        
        self.logger.info("Conversation Metrics",
            phase=phase,
            status=status,
            language=language,
            phase_stats=self.metrics["conversation"][f"{phase}_phase"]
        )


monitoring = ChatbotMonitoring()