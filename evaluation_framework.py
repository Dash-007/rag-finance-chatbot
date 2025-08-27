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
    
class ABTestManager:
    """
    A/B testing manafer for comparing different RAG configurations.
    """
    
    def __init__(self, evaluator: RAGEvaluator):
        self.evaluator = evaluator
        self.experiments = {}
        
    async def run_ab_test(self,
                          model_a,
                          model_b,
                          test_name: str,
                          model_a_name: str = "baseline",
                          model_b_name: str = "variant") -> Dict[str, Any]:
        """
        Run A/B test comparing two model configurations.
        
        Args:
            model_a: Baseline model instance
            model_b: Variant model instance
            test_name: Name for this experiment
            model_a_name: Name for model A
            model_b_name: Name for model B
            
        Returns:
            Comparision results
        """
        print(f"🧪 Starting A/B Test: {test_name}")
        print(f"Model A: {model_a_name}")
        print(f"Model B: {model_b_name}")
        
        # Evaluate both models
        results_a = await self.evaluator.evaluate_model(model_a, model_a_name)
        results_b = await self.evaluator.evaluate_model(model_b, model_b_name)
        
        # Calculate aggregate metrics
        comparison = self._compare_results(results_a, results_b, model_a_name, model_b_name)
        
        # Save experiment data
        experiment_data = {
            'test_name': test_name,
            'timestamp': datetime.now().isoformat(),
            'results_a': results_a,
            'results_b': results_b,
            'comparison': comparison
        }
        
        self.experiments[test_name] = experiment_data
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.evaluator.save_results(results_a, f"ab_test_{test_name}_{model_a_name}_{timestamp}.json")
        self.evaluator.save_results(results_b, f"ab_test_{test_name}_{model_b_name}_{timestamp}.json")
        
        return comparison
    
    def _compare_results(self,
                         results_a: List[EvaluationResult],
                         results_b: List[EvaluationResult],
                         name_a: str,
                         name_b: str) -> Dict[str, Any]:
        """
        Compare results from two model variants.
        """
        
        def aggregate_metrics(results: List[EvaluationResult]) -> Dict[str, float]:
            """
            Calculate average metrics across all results.
            """
            if not results:
                return {}
            
            metric_sums = {}
            for result in results:
                for metric, value in result.metric.items():
                    metric_sums[metric] = metric_sums.get(metric, 0) + value
                    
            return {metric: total / len(results) for metric, total in metric_sums.items()}
        
        metrics_a = aggregate_metrics(results_a)
        metrics_b = aggregate_metrics(results_b)
        
        # Calculate improvements
        improvements = {}
        for metric in metrics_a:
            if metric in metrics_b:
                diff = metrics_b[metric] - metrics_a[metric]
                improvements[metric] = {
                    'absolute_diff': diff,
                    'percentage_diff': (diff / metrics_a[metric] * 100) if metrics_a[metric] > 0 else 0
                }
                
        # Average response times
        avg_time_a = sum(r.response_time for r in results_a) / len(results_a)
        avg_time_b = sum(r.response_time for r in results_b) / len(results_b)
        
        return {
            'model_a': {
                'name': name_a,
                'metrics': metrics_a,
                'avg_response_time': avg_time_a
            },
            'model_b': {
                'name': name_b,
                'metrics': metrics_b,
                'avg_response_time': avg_time_b
            },
            'improvements': improvements,
            'winner': self._determine_winner(metrics_a, metrics_b, name_a, name_b)
        }
        
    def _determine_winner(self, metrics_a: Dict, metrics_b: Dict, name_a: str, name_b: str) -> str:
        """
        Simple winner determination based on average metric scores.
        """
        avg_a = sum(metrics_a.values()) / len(metrics_a) if metrics_a else 0
        avg_b = sum(metrics_b.values()) / len(metrics_b) if metrics_b else 0
        
        if avg_b > avg_a:
            return name_b
        elif avg_a > avg_b:
            return name_a
        else:
            return "tie"
        
    def get_experiment_summary(self, test_name: str) -> Dict[str, Any]:
        """
        get summary of a specific experiment.
        """
        if test_name not in self.experiments:
            return {"Error": f"Experiment '{test_name}' not found!"}
        
        return self.experiments[test_name]['comparison']