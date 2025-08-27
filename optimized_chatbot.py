import os
import asyncio
import logging
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from config_manager import ConfigManager, ModelConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class QueryType(Enum):
    """Financial query categories for specialized handling.
    """
    DEFINITION = "definition"
    CALCULATION = "calculation"
    ADVICE = "advice"
    COMPARISON = "comparison"
    GENERAL = "general"
    
@dataclass
class QueryContext:
    """
    Query classification context.
    """
    query_type: QueryType
    confidence: float

class QueryClassifier:
    """
    Improved query classifier with better pattern matching.
    """
    
    def __init__(self):
        self.patterns = {
            QueryType.DEFINITION: {
                "keywords": ["what is", "define", "meaning", "explain", "term"],
                "weight": 1.0
            },
            QueryType.CALCULATION: {
                "keywords": ["calculate", "formula", "how much", "rate", "percentage", "compute"],
                "weight": 1.0
            },
            QueryType.ADVICE: {
                "keywords": ["should i", "recommend", "better", "best", "strategy", "advice"],
                "weight": 0.9
            },
            QueryType.COMPARISON: {
                "keywords": ["vs", "versus", "compare", "difference", "between"],
                "weight": 1.0
            }
        }
        
    def classify(self, query: str) -> QueryContext:
        """
        Classify query with improved confidence score.
        """
        query_lower = query.lower()
        
        scores = {}
        for query_type, pattern_data in self.patterns.items():
            score = 0
            keywords = pattern_data["keywords"]
            weight = pattern_data["weight"]
            
        # Exact phrase matching
        for keyword in keywords:
            if keyword in query_lower:
                score += 0.4 * weight
                
        # Word matching
        query_words = query_lower.split()
        for keyword in keyword:
            keyword_words = keyword.split()
            if any(word in query_words for word in keyword_words):
                score += 0.2 * weight
                
        scores[query_type] = min(score, 1.0)
        
        # Default general score
        scores[QueryType.GENERAL] = 0.3
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        return QueryContext(query_type=best_type, confidence=confidence)
    
class AdvancedRetriever:
    """
    Optimized retrieval system with error handling and caching.
    """
    
    def __init__(self, vectore_store: PineconeVectorStore, config: ModelConfig):
        self.vector_store = vectore_store
        self.config = config
        self.cache = {} # In-memory cache
        
    async def retrieve(self, query: str, query_context: QueryContext) -> List[Document]
        """
        Retrieve documents with fallback strategies.
        """
        k = self.config.retrieval_k
        cache_key = f"{query}_{query_context.query_type.value}_{k}"
        
        # Check cache first
        if cache_key in self.cache:
            logger.info(f"Cache hit for query: {query[:50]}...")
            return self.cache[cache_key]
        
        try:
            if query_context.query_type == QueryType.DEFINITION:
                docs = await self._definition_retrieval(query, k)
            elif query_context.query_type == QueryType.CALCULATION:
                docs = await self._calculation_retrieval(query, k)
            elif query_context.query_type == QueryType.COMPARISON:
                docs = await self._comparison_retrieval(query, k)
            else:
                docs = await self._standard_retrieval(query, k)
                
            # Cache successful results
            self.cache[cache_key] = docs
            return docs
        
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return await self._standard_retrieval(query, k) # Fallback to standard retrieval
        
    