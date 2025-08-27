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
        
    async def retrieve(self, query: str, query_context: QueryContext) -> List[Document]:
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
        
    async def _definition_retrieval(self, query: str, k: int) -> List[Document]:
        """
        Enhanced definition-focused retrieval.
        """
        try:
            docs = self.vector_store.similarity_search_with_score(query, k=k*2)
            
            definition_docs = []
            standard_docs = []
            
            for doc, score in docs:
                content_lower = doc.page_content.lower()
                definition_indicators = [
                    'definition', 'means', 'refers to', 'is a', 'is the',
                    'defined as', 'can be described as', 'is when'
                ]
                
                if any(indicator in content_lower for indicator in definition_indicators):
                    doc.metadata['strategy'] = 'definition_focused'
                    doc.metadata['relevance_score'] = score
                    definition_docs.append(doc)
                else:
                    standard_docs.append((doc, score))
                    
            # Fill remaining slots with highest scoring standard docs
            while len(definition_docs) < k and standard_docs:
                doc, score = standard_docs.pop(0)
                doc.metadata['stratedy'] = 'standard'
                doc.metadata['relevance_score'] = score
                definition_docs.append(doc)
                
            return definition_docs[:k]
        
        except Exception as e:
            logger.error(f"Definition retrieval error: {e}")
            return await self._standard_retrieval(query, k)
        
    async def _calculation_retrieval(self, query: str, k: int) -> List[Document]:
        """
        Enhanced calculation-focused retrieval.
        """
        try:
            docs = self.vector_store.similarity_search_with_score(query, k=k*2)
            
            calc_docs = []
            standard_docs = []
            
            for doc, score in docs:
                content_lower = doc.page_content.lower()
                calc_indicators = {
                    'formula', 'calculate', '=', '%', 'example', 'step',
                    'equation', 'method', 'how to', 'multiply', 'divide'
                }
                
                calc_score = sum(1 for indicator in calc_indicators if indicator in content_lower)
                
                if calc_score >= 2:
                    doc.metadata['strategy'] = 'calculation_focused'
                    doc.metadata['relevance_score'] = score
                    doc.metadata['calc_score'] = calc_score
                    calc_docs.append(doc)
                else:
                    standard_docs.append((doc, score))
                    
            # Sort calc docs by calculation score
            calc_docs.sort(key=lambda x: x.metadata.get('calc_score', 0), reverse=True)
            
            # Fill remaining slots
            while len(calc_docs) < k and standard_docs:
                doc, score = standard_docs.pop(0)
                doc.metadata['strategy'] = 'standard'
                doc.metadata['relevance_score'] = score
                calc_docs.append(doc)
                
            return calc_docs[:k]
        
        except Exception as e:
            logger.error(f"Calculation retrieval error: {e}")
            return await self._standard_retrieval(query, k)
        
    async def _comparison_retrieval(self, query:str, k: int) -> List[Document]:
        """
        Enhanced comparison retrieval with term extraction.
        """
        try:
            query_lower = query.lower()
            terms = []
            
            delimiters = ['vs ', 'versus ', ' or ', 'between ', ' and ']
            for delimiter in delimiters:
                if delimiter in query_lower:
                    parts = query_lower.split(delimiter, 1)
                    if len(parts) == 2:
                        terms = [part.strip() for part in parts]
                        break
                    
            if not terms:
                # Fallback: look for common comparison words
                if any(word in query_lower for word in ['compare', 'difference', 'better']):
                    terms = [query_lower]
                else:
                    return await self._standard_retrieval(query, k)
                
            all_docs = []
            for term in terms[:2]:
                try:
                    docs = self.vector_store.similarity_search_with_score(term, k=k//2+1)
                    for doc, score in docs:
                        doc.metadata['strategy'] = 'comparative'
                        doc.metadata['search_term'] = term
                        doc.metadata['relevance_score'] = score
                        all_docs.append(doc)
                    
                except Exception as e:
                    logger.warning(f"Error searching for term '{term}': {e}")
                    
            # Remove duplicates and return top k
            seen_content = set()
            unique_docs = []
            for doc in all_docs:
                content_key = doc.page_content[:100]
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    unique_docs.append(doc)
                    
            return unique_docs[:k]
        
        except Exception as e:
            logger.error(f"Comparison retrieval error: {e}")
            return await self._standard_retrieval(query, k)
        
    async def _standard_retrieval(self, query: str, k: int) -> List[Document]:
        """
        Fallback standard retrieval.
        """
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            for doc in docs:
                doc.metadata['strategy'] = 'standard'
            return docs
        except Exception as e:
            logger.error(f"Standard retrieval error: {e}")
            return []