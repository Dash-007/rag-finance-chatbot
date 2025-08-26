import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
from pathlib import Path

class EvaluationMetric(Enum):
    """
    Evaluation metrics for RAG system performance.
    """
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    RESPONSE_TIME = "response_time"
    USER_SATISFACTION = "user_satisfaction"
    
@dataclass
class TestQuery:
    """
    Test query with expected characteristics.
    """
    query: str
    query_type: str
    expected_keywords: List[str]
    difficulty: str
    ground_truth: Optional[str] = None
    
@dataclass
class EvaluationResult:
    """
    Single evaluation result.
    """
    query_id: str
    query: str
    response: str
    response_time: float
    query_type: str
    model_version: str
    timestamp: datetime
    metrics: Dict[str, float]
    sources_used: int
    
class RAGEvaluator:
    """
    Evaluation framework for RAG chatbot with A/B testing cababilities.
    """
    
    def __init__(self, results_dir: str = "evaluation_results"):
        self.result_dir = Path(results_dir)
        self.result_dir.mkdir(exist_ok=True)
        self.test_queries = self._load_test_queries()
        
    def _load_test_queries(self) -> List[TestQuery]:
        """
        Load predefined test queries for consistent evaluation.
        """
        return [
            TestQuery(
                query="What is compound interest?",
                query_type="definition",
                expected_keywords=["interest", "compound", "principal", "time"],
                difficulty="easy"
            ),
            TestQuery(
                query="How do I calculate the present value of an annuity?",
                query_type="calculation", 
                expected_keywords=["present value", "annuity", "formula", "discount rate"],
                difficulty="medium"
            ),
            TestQuery(
                query="Should I invest in growth stocks or value stocks for retirement?",
                query_type="advice",
                expected_keywords=["growth", "value", "retirement", "risk", "diversification"],
                difficulty="hard"
            ),
            TestQuery(
                query="Compare 401k vs IRA for tax advantages",
                query_type="comparison",
                expected_keywords=["401k", "IRA", "tax", "contribution limits", "benefits"],
                difficulty="medium"
            ),
            TestQuery(
                query="What factors affect mortgage interest rates?",
                query_type="general",
                expected_keywords=["mortgage", "interest rates", "credit score", "economic factors"],
                difficulty="easy"
            )
        ]
        
    def _calculate_metrics(self, test_query: TestQuery, response: str, response_time: float) -> Dict[str, float]:
        """
        Calculate evaluation metrics for a single query-response pair.
        
        Args:
            test_query: Test query object
            response: Model response
            response_time: Response time in seconds
        
        Returns:
            Dictionary of metric scores (0-1 scale)
        """
        response_lower = response.lower()
        
        # Keyword coverage
        keyword_matches = sum(1 for kw in test_query.expected_keywords
                              if kw.lower() in response_lower)
        keyword_coverage = keyword_matches / len(test_query.expected_keywords)
        
        # Response completeness
        completeness = min(len(response) / 200, 1.0)
        
        # Response time score
        time_score = max(0, 1 - (response_time / 10))
        
        return {
            EvaluationMetric.RELEVANCE.value: keyword_coverage,
            EvaluationMetric.COMPLETENESS.value: completeness,
            EvaluationMetric.RESPONSE_TIME.value: time_score,
            EvaluationMetric.ACCURACY.value: 0.8, # Placeholder
        }
        
    async def evaluate_model(self, bot, model_version: str = "v1") -> List[EvaluationResult]:
        """
        Evaluate chatbot performance on test queries.
        
        Args:
            chatbot: RAG chatbot instance
            model_version: Version identifier for A/B testing
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, test_query in enumerate(self.test_queries):
            print(f"Evaluation query {i+1}/{len(self.test_queries)}: {test_query.query}")
            
            # Measure response time
            start_time = time.time()
            response_data = await bot.ask(test_query.query)
            end_time = time.time()
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                test_query,
                response_data.get('response', ''),
                end_time - start_time
            )
            
            # Create evaluation report
            result = EvaluationResult(
                query_id = f"q_{i+1}",
                query = test_query.query,
                response = response_data.get('response', ''),
                response_time = end_time - start_time,
                query_type=test_query.query_type,
                model_version=model_version,
                timestamp=datetime.now(),
                metrics=metrics,
                sources_used=response_data.get('sources', 0)
            )
            
            results.append(result)
            
        return results
    
    def save_results(self, results: List[EvaluationResult], filename: str = None):
        """
        Save evaluation results to JSON file.
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_results_{timestamp}.json"
            
        filepath = self.result_dir / filename
        
        # Convert results to serializable format
        serializable_result = []
        for result in results:
            result_dict = asdict(result)
            result_dict['timestamp'] = result.timestamp.isoformat()
            serializable_result.append(result_dict)
            
        with open(filepath, 'w') as f:
            json.dump(serializable_result, f, indent=2)
            
        print(f"Results saved to {filepath}")
        
        return filepath
    
    def load_results(self, filename: str) -> List[EvaluationResult]:
        """
        Load evaluation results from JSON file.
        """
        filepath = self.result_dir / filename
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        results = []
        for item in data:
            item['timestamp'] = datetime.fromisoformat(item['timestamp'])
            results.append(EvaluationResult(**item))
            
        return results