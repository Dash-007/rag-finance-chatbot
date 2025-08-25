import os
import asyncio
from enum import Enum
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from langchain.prompts import PromptTemplate

load_dotenv()

class QueryType(Enum):
    """
    Financial query categories for specialized handling.
    """
    DEFINITION = "definition"
    CALCULATION = "calculation"
    ADVICE = "advice"
    COMPARISON = "comparison"
    GENERAL = "general"
    
@dataclass
class QueryContext:
    query_type: QueryType
    confidence: float
    
class QueryClassifier:
    """
    Classifies queries to determine optimal retrieval strategy.
    """
    
    def __init__(self):
        # Keyword patterns for query classification
        self.patterns = {
            QueryType.DEFINITION: ["what is", "define", "meaning","explain","term"],
            QueryType.CALCULATION: ["calculate", "formula", "how much", "rate", "percentage"],
            QueryType.ADVICE: ["should i", "recommend", "better", "best", "strategy"],
            QueryType.COMPARISON: ["vs", "versus", "compare", "difference", "between"]
        }
        
        def classify(self, query: str) -> QueryContext:
            """
            Analyze query and return classification with confidence.
            """
            query_lower = query.lower()
            
            # Calculate scores for each query type
            scores = {}
            for query_type, keywords in self.patterns.items():
                score = sum(0.3 for keyword in keywords if keyword in query_lower)
                scores[query_type] = min(score, 1.0)
                
            # Add baseline for general queries
            scores[QueryType.GENERAL] = 0.2
            
            # Get best match
            best_type =max(scores, key=scores.get)
            confidence = scores[best_type]
            
            return QueryContext(query_type=best_type, confidence=confidence)
        
    