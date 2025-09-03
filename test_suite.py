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
        
