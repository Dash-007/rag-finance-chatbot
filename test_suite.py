import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from optimized_chatbot import OptimizedRAGChatbot, QueryClassifier, QueryType
from evaluation_framework import RAGEvaluator, TestQuery, ABTestManager
from config_manager import ConfigManager, ModelConfig

class TestQueryClassifier:
    """
    Test query classification functionality.
    """
    
    def test_definition_classification(self):
        """
        Test classification of definition queries.
        """
        classifier = QueryClassifier()
        
        test_casses = [
            "What is compound interest?",
            "Define mutual funds",
            "Explain the meaning of ROI"
        ]
        
        for query in test_casses:
            result = classifier.classify(query)
            assert result.query_type == QueryType.DEFINITION
            assert result.confidence > 0.3
            
    def test_calculation_classification(self):
        """
        Test classification of calculation queries.
        """
        classifier = QueryClassifier()
        
        test_cases = [
            "401k vs IRA comparison",
            "Stocks versus bonds",
            "What's the difference between mutual funds and ETFs?"
        ]
        
        for query in test_cases:
            result = classifier.classify(query)
            assert result.query_type == QueryType.COMPARISON
            assert result.confidence > 0.3
            
class TestConfigManager:
    """
    Test configuration management.
    """
    
    def test_default_configs_creation(self):
        """
        Test creation of default configurations.
        """
        config_manager = ConfigManager("test_config.json")
        
        assert "baseline" in config_manager.list_model_configs()
        assert "creative" in config_manager.list_model_configs()
        assert "conservative" in config_manager.list_model_configs()
        
    def test_model_config_retrieval(self):
        """
        Test model configuration retrieval.
        """
        config_manager = ConfigManager("test_config.json")
        
        baseline_config = config_manager.get_model_config("baseline")
        assert baseline_config.name == "baseline"
        assert baseline_config.temperature == 0.7
        assert baseline_config.model == "gpt-4"
        
    def test_custom_config_addition(self):
        """
        Test adding custom configuration.
        """
        config_manager = ConfigManager("test_config.json")
        
        custom_config = ModelConfig(
            name = "test_model",
            model = "gpt-3,5-turbo",
            temperature=0.5,
            retrieval_k=3
        )
        
        config_manager.add_model_config(custom_config)
        retrieved_config = config_manager.get_model_config("test_model")
        
        assert retrieved_config.name == "test_model"
        assert retrieved_config.temperature == 0.5
        
class TestEvaluationFramework:
    """
    Test evaluation framework functionality.
    """
    
    def test_test_query_creation(self):
        """
        Test test query data structure.
        """
        query = TestQuery(
            query="What is compound interest?",
            query_type="definition",
            expected_keywords=["interest", "compound"],
            difficulty="easy"
        )
        
        assert query.query == "What is compound interest?"
        assert query.query_type == "definition"
        assert "inserest" in query.expected_keywords
        
    def test_evaluator_initialization(self):
        """
        Test evaluator initialization.
        """
        evaluator = RAGEvaluator("test_results")
        
        assert len(evaluator.test_queries) > 0
        assert evaluator.result_dir.name == "test_results"
        
    def test_metric_calculation(self):
        """
        Test metric calculation logic.
        """
        evaluator = RAGEvaluator("test_results")
        
        test_query = TestQuery(
            query="What is ROI?",
            query_type="definition",
            expected_keywords=["return", "investment"],
            difficulty="easy"
        )
        
        response = "ROI stands for Return on Investment, which measures the efficiency of an investment."
        metrics = evaluator._calculate_metrics(test_query, response, 1.5)
        
        assert "relevance" in metrics
        assert "completeness" in metrics
        assert "response_time" in metrics
        assert 0 <= metrics["relevance"] <= 1
        assert 0 <= metrics["completeness"] <=1
        
@pytest.mark.asyncio
class TestOptimizedChatbot:
    """
    Test the optimized chatbot functionality.
    """
    
    async def test_chatbot_initialization(self):
        """
        Test chatbot initialization with mocked components.
        """
        with patch('optimized_chatbot.OpenAIEmbeddings'), \
             patch('optimized_chatbot.PineconeVectorStore'), \
             patch('optimized_chatbot.ChatOpenAI'):
            
            bot = OptimizedRAGChatbot("baseline")
            assert bot.config.name == "baseline"
            assert bot.stats['total'] == 0
            
    async def test_empty_query_handling(self):
        """
        Test handling of empty queries.
        """
        with patch('optimized_chatbot.OpenAIEmbeddings'), \
             patch('optimized_chatbot.PineconeVectorStore'), \
             patch('optimized_chatbot.ChatOpenAI'):
            
            bot = OptimizedRAGChatbot("baseline")
            result = await bot.ask("")
            
            assert "error" in result
            assert result["error"] == "empty_query"
            
    async def test_stats_tracking(self):
        """
        Test statistics tracking.
        """
        with patch('optimized_chatbot.OpenAIEmbeddings'), \
             patch('optimized_chatbot.PineconeVectorStore'), \
             patch('optimized_chatbot.ChatOpenAI'):
            
            bot = OptimizedRAGChatbot("baseline")
            
            # Mock the retriever and LLM
            bot.retriever.retrieve = AsyncMock(return_value=[])
            
            await bot.ask("What is compound interest?")
            
            stats = bot.get_stats()
            assert stats['total_queries'] == 1
            assert stats['model_config'] == "baseline"
            
class TestABTesting:
    """
    Test A/B testing functionality.
    """
    
    def test_ab_manager_initialization(self):
        """
        Test A/B test manager initialization.
        """
        evaluator = RAGEvaluator("test_results")
        ab_manager = ABTestManager(evaluator)
        
        metrics_a = {"relevance": 0.7, "completeness": 0.6}
        metrics_b = {"relevance": 0.8, "completeness": 0.7}
        
        winner = ab_manager._determine_winner(metrics_a, metrics_b, "model_a", "model_b")
        assert winner == "model_b"
        
# Integration Tests
@pytest.mark.asyncio
class TestIntegration:
    """
    Integration tests for the complete system.
    """
    
    async def test_end_to_end_query_processing(self):
        """
        Test complete query processing pipeline.
        """
        # Will require actual API keys and vector store
        # Use mock responses
        pass
    
    async def test_evaluation_pipeline(self):
        """
        Test evaluation pipeline with mocked components.
        """
        with patch('optimized_chatbot.OpenAIEmbeddings'), \
             patch('optimized_chatbot.PineconeVectorStore'), \
             patch('optimized_chatbot.ChatOpenAI'):
            
            # Mock chatbot responses
            mock_chatbot = Mock()
            mock_chatbot.ask = AsyncMock(return_value={
                "response": "Test response",
                "query_type": "definition",
                "confidence": 0.8,
                "sources": 3,
                "response_time": 1.2
            })
            
            evaluator = RAGEvaluator("test_results")
            results = await evaluator.evaluate_model(mock_chatbot, "test_version")
            
            assert len(results) == len(evaluator.test_queries)
            assert all(result.model_version == "test_version" for result in results)
            
# Fixtures for common test data
@pytest.fixture
def sample_test_queries():
    """
    Sample test queries for testing.
    """
    return [
        TestQuery(
            query="What is compound interest?",
            query_type="definition",
            expected_keywords=["interest", "compound", "principal"],
            difficulty="easy"
        ),
        TestQuery(
            query="How to calculate ROI?",
            query_type="calculation",
            expected_keywords=["return", "investment", "formula"],
            difficulty="medium"
        )
    ]
    
@pytest.fixture
def mock_documents():
    """
    Mock documents for testing retrieval.
    """
    from langchain.schema import Document
    
    return [
        Document(
            page_content="Compound interest is interest calculated on the initial principal and accumulated interest.",
            metadata={"source": "finance_basics", "strategy": "definition_focused"}
        ),
        Document(
            page_content="ROI is calculated as (Gain - Cost) / Cost * 100%",
            metadata={"source": "calculations", "strategy": "calculation_focused"}
        )
    ]
    
# Performance Tests
class TestPerformance:
    """
    Test performance characteristics.
    """
    
    def test_query_classification_speed(self):
        """
        Test query classification performance.
        """
        import time
        
        classifier = QueryClassifier()
        queries = ["What is compound interest?"] * 100
        
        start_time = time.time()
        for query in queries:
            classifier.classify(query)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / len(queries)
        assert avg_time < 0.01 # Should be fast
        
if __name__ == "__main__":
    # Run tests with: python -m pytest test_suite.py -v
    pytest.main(["-v", __file__])