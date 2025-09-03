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
            
