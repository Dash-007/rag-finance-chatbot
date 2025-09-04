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
from src.config_manager import ConfigManager, ModelConfig

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
            for keyword in keywords:
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
    
    def __init__(self, vector_store: PineconeVectorStore, config: ModelConfig):
        self.vector_store = vector_store
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
                doc.metadata['strategy'] = 'standard'
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
                calc_indicators = [
                    'formula', 'calculate', '=', '%', 'example', 'step',
                    'equation', 'method', 'how to', 'multiply', 'divide'
                ]
                
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
        
class OptimizedRAGChatbot:
    """
    Production-ready RAG chatbot with configuration management and monitoring.
    """
    
    def __init__(self, config_name: str = "baseline"):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_model_config(config_name)
        self.chat_history = []
        self.stats = {'total': 0, 'types': {qt.value: 0 for qt in QueryType}}
        
        # Initialize components with error handling
        try:
            self._initialize_components()
            logger.info(f"Initialized chatbot with config: {config_name}")
        
        except Exception as e:
            logger.error(f"Failed to initialize chatbot: {e}")
            raise
        
    def _initialize_components(self):
        """
        Initialize LangChain components.
        """
        # Embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model,
            dimensions=self.config.embedding_dimensions,
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Vector store
        self.vector_store = PineconeVectorStore(
            index_name=os.environ.get("INDEX_NAME"),
            embedding=self.embeddings
        )
        
        # LLM
        llm_kwargs = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "openai_api_key": os.environ.get("OPENAI_API_KEY")
        }
        if self.config.max_tokens:
            llm_kwargs["max_tokens"] = self.config.max_tokens
            
        self.llm = ChatOpenAI(**llm_kwargs)
        
        # Components
        self.classifier = QueryClassifier()
        self.retriever = AdvancedRetriever(self.vector_store, self.config)
        
    async def ask(self, question: str) -> Dict[str, Any]:
        """
        Process question with comprehensive error handling and monitoring.
        """
        start_time = asyncio.get_event_loop().time()
        self.stats['total'] += 1
        
        try:
            # Input validation
            if not question or not question.strip():
                return {
                    "response": "Please provide a valid question.",
                    "error": "empty_query"
                }
                
            # Classify query
            context = self.classifier.classify(question)
            self.stats['types'][context.query_type.value] += 1
            
            logger.info(f"Classified query as: {context.query_type.value}"
                        f"(confidence: {context.confidence:.2f})")
            
            # Retrieve documents
            docs = await self.retriever.retrieve(question, context)
            
            if not docs:
                logger.warning(f"No documents retrieved!")
                return {
                    "response": "I couldn't find relevant information to answer your question. "
                        "Please try rephrasing or asking about a different topic.",
                    "query_type": context.query_type.value,
                    "confidence": context.confidence,
                    "sources": 0
                }
                
            # Generate response
            prompt = self._create_prompt(context.query_type)
            context_text = self._format_context(docs)
            history_text = self._format_history()
            
            full_prompt = f"""{prompt}
            
            Context: {context_text}
            
            History: {history_text}
            
            Question: {question}
            
            Response: """
            
            try:
                response = self.llm.invoke([{"role": "user", "content": full_prompt}])
                answer = response.content
                
            except Exception as e:
                logger.error(f"LLM generation error: {e}")
                return {
                    "response": "I encountered an issue generating a response. Please try again.",
                    "error": "generation_error",
                    "query_type": context.query_type.value
                }
                
            # Update history
            self.chat_history.append((question, answer))
            # Manage history
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-5:]
                
            # Calculate response time
            response_time = asyncio.get_event_loop().time() - start_time
            
            return {
                "response": answer,
                "query_type": context.query_type.value,
                "confidence": context.confidence,
                "sources": len(docs),
                "response_time": response_time,
                "model_config": self.config.name
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in ask(): {e}")
            return {
                "response": "I encountered an unexpected issue. Please try again.",
                "error": str(e),
                "query_type": "unknown"
            }
            
    def _create_prompt(self, query_type: QueryType) -> str:
        """
        Create specialized prompts with better instructions.
        """
        base = ("You are a knowledgeable financial assistant. "
                "Provide accurate, helpful, and well-structured responses based on the context provided.")
        
        prompts = {
            QueryType.DEFINITION: f"{base} Focus on clear, comprehensive definitions with practical examples.",
            QueryType.CALCULATION: f"{base} Provide step-by-step calculations with formulas and worked examples.",
            QueryType.ADVICE: f"{base} Give balanced, practical advice. Always recommend consulting with financial professionals for personalized advice.",
            QueryType.COMPARISON: f"{base} Compare options objectively, highlighting key differences, pros, and cons.",
            QueryType.GENERAL: f"{base} Provide comprehensive, well-organized financial guidance."
        }
        
        return prompts.get(query_type, prompts[QueryType.GENERAL])
            
    def _format_context(self, docs: List[Document]) -> str:
        """
        Format retrieved documents with metadata.
        """
        if not docs:
            return "No relevant context found."
        
        formatted_parts = []
        for i, doc in enumerate(docs, 1):
            strategy = doc.metadata.get('strategy', 'unknown')
            content = doc.page_content[:500]
            formatted_parts.append(f"[Source {i} - {strategy}]: {content}")
            
        return "\n\n".join(formatted_parts)
    
    def _format_history(self) -> str:
        """
        Format conversation history efficiently.
        """
        if not self.chat_history:
            return "No previous conversation."
        
        # Only include last 2 exchanges to manage prompt length
        recent = self.chat_history[-2:]
        history_parts = []
        for q, a in recent:
            # Truncate long responses
            truncated_a = a[:200] + "..." if len(a) > 200 else a
            history_parts.append(f"Q: {q}\nA: {truncated_a}")
            
        return "\n".join(history_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive session statistics.
        """
        return {
            'total_queries': self.stats['total'],
            'query_types': self.stats['types'].copy(),
            'model_config': self.config.name,
            'history_length': len(self.chat_history),
            'cache_size': len(self.retriever.cache)
        }
        
    def clear_history(self):
        """
        Clear conversation history and cache.
        """
        self.chat_history = []
        self.retriever.cache = {}
        logger.info(f"Cleared history and cache")
        
    def get_config_name(self):
        """
        Get current configuration name.
        """
        return self.config.name
    
# Factory function for easy installation
def create_chatbot(config_name: str = "baseline") -> OptimizedRAGChatbot:
    """
    Factory function to create a chatbot with specified configuration.
    
    Args:
        config_name: Name of the configuration to use
        
    Returns:
        Configured OptimizedRAGChatbot instance
    """
    return OptimizedRAGChatbot(config_name)

# Main execution functions for testing
async def demo():
    """
    Quick demo of the optimized system.
    """
    print("🚀 Optimized RAG Chatbot Demo\n")
    
    bot = create_chatbot("baseline")
    
    queries = [
        "What is compound interest?",
        "How do I calculate ROI on investment?", 
        "Should I invest in stocks or bonds?",
        "What's the difference between 401k and IRA?"
    ]
    
    for query in queries:
        print(f"Query: {query}")
        result = await bot.ask(query)
        print(f"Response: {result['response'][:100]}...")
        print(f"Type: {result.get('query_type', 'unknown')}, "
              f"Confidence: {result.get('confidence', 0):.2f}, "
              f"Time: {result.get('response_time', 0):.2f}s\n")
        print("-" * 50)
        
if __name__ == "__main__":
    asyncio.run(demo())