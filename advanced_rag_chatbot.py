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
        
class AdvancedRetriever:
    """
    Retrieval ststem with se=pecialized strategies based on query type.
    """
    def __init__(self, vector_store: PineconeVectorStore):
        self.vectore_store = vector_store
        
    async def retrieve(self, query: str, query_context: QueryContext, k: int = 4) -> List[Document]:
        """
        Perform specialized retrieval based on query classification.
        """
        if query_context.query_type == QueryType.DEFINITION:
            return await self._definition_retrieval(query, k)
        elif query_context.query_type == QueryType.CALCULATION:
            return await self._calculation_retrieval(query, k)
        elif query_context.query_type == QueryType.COMPARISON:
            return await self._comparison_retrieval(query, k)
        else:
            return await self._standard_retrieval(query, k)
        
    async def _definition_retrieval(self, query: str, k: int) -> List[Document]:
        """
        Prioritize definition-rich content.
        """
        docs = self.vectore_store.similarity_search_with_score(query, k=k*2)
        
        # Filter for definition indicators
        definition_docs = []
        for doc, score in docs:
            content = doc.page_content.lower()
            if any(term in content for term in ['definition', 'means', 'refers to', 'is a']):
                doc.metadata['strategy'] = 'definition_focused'
                definition_docs.append(doc)
                
        # Fill remaining slots with regular results
        if len(definition_docs) < k:
            for doc, score in docs:
                if doc not in definition_docs:
                    doc.metadata['strategy'] = 'standard'
                    definition_docs.append(doc)
                    if len(definition_docs) >= k:
                        break
                    
        return definition_docs[:k]
    
    async def _calculation_retrieval(self, query: str, k: int) -> List[Document]:
        """
        Prioritize content with formilas and examples.
        """
        docs = self.vectore_store.similarity_search_with_score(query, k=k*2)
        
        # Filter for calculation indications
        calc_docs = []
        for doc, score in docs:
            content = doc.page_content.lower()
            if any(term in content for term in ['formula', 'calculate', '=', '%', 'example']):
                doc.metadata['strategy'] = 'calculation_focused'
                calc_docs.append(doc)
                
        # Fill remaining slots
        if len(calc_docs) < k:
            for doc, score in docs:
                if doc not in calc_docs:
                    doc.metadata['strategy'] = 'standard'
                    calc_docs.append(doc)
                    if len(calc_docs) >= k:
                        break
                    
        return calc_docs[:k]
    
    async def _comparison_retrieval(self, query: str, k: int) -> List[Document]:
        """
        Get diverse content for comparison queries.
        """
        query_lower = query.lower()
        terms = []
        
        for delimiter in ['vs ', 'versus ', 'or ', 'between ']:
            if delimiter in query_lower:
                parts = query_lower.split(delimiter)
                terms.extend([part.strip() for part in parts[:2]])
                break
            
        if not terms:
            terms = [query_lower]
            
        # Search for each term
        all_docs = []
        for term in terms [:2]:
            docs = self.vectore_store.similarity_search(term, k=k//2+1)
            for doc in docs:
                doc.metadata['strategy'] = 'comparative'
            all_docs.extend(docs)
            
        # Remove duplicate and return top k
        unique_docs = {doc.page_content[:50]: doc for doc in all_docs}
        return list(unique_docs.values())[:k]
    
    async def _standard_retrieval(self, query: str, k: int) -> List[Document]:
        """
        Standard semantic search.
        """
        docs = self.vectore_store.similarity_search(query, k=k)
        for doc in docs:
            doc.metadata['strategy'] = 'standard'
        return docs
    
class AdvancedRAGChatbot:
    """
    Advanced RAG system with query classification and specialized retrieval.
    """
    
    def __init__(self, model="gpt-4", temperature=0.7):
        self.chat_history = []
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=512,
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        self.vector_store = PineconeVectorStore(
            index_name = os.environ.get("INDEX_NAME"),
            embedding=self.embeddings
        )
        
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Advanced components
        self.classifier = QueryClassifier()
        self.retriever = AdvancedRetriever(self.vector_store)
        
        # Simple stats tracking
        self.stats = {'total': 0, 'types': {qt.value: 0 for qt in QueryType}}
        
    async def ask(self, question: str) -> Dict[str, Any]:
        """
        Process question using advanced RAG pipeline.
        """
        self.stats['total'] += 1
        
        try:
            # Step 1: Classify query
            context = self.classifier.classify(question)
            self.stats['types'][context.query_type.value] += 1
            
            print(f"Classified as: {context.query_type.value} (confidence: {context.confidence:.2f})")
            
            # Step 2: Specialized retrieval
            docs = await self.retriever.retrieve(question, context)
            
            # Step 3: Generate response with specialized prompt
            prompt = self._create_prompt(context.query_type)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            history_text = self._format_history()
            
            # Build final prompt
            full_prompt = f"""{prompt}
            
        Context: {context_text}
        
        History: {history_text}
        
        Question: {question}
        
        Response:"""
        
            # Generate answer
            response = self.llm.invoke([{"role": "user", "content": full_prompt}])
            answer = response.content
            
            # Update history
            self.chat_history.append((question, answer))
            
            return {
                "response": answer,
                "query_type": context.query_type.value,
                "confidence": context.confidence,
                "sources": len(docs)
            }
            
        except Exception as e:
            print(f"Error: {e}")
            return {
                "response": "I encountered an issue. Please try again.",
                "error": str(e)
            }
            
    def _create_prompt(self, query_type: QueryType) -> str:
        """
        Create specialized prompts for different query types.
        """
        base = "You are a knowledgeable financial assistant."
        
        prompts = {
            QueryType.DEFINITION: f"{base} Focus on clear definitions with examples.",
            QueryType.CALCULATION: f"{base} Provide step-by-step calculations and formulas.",
            QueryType.ADVICE: f"{base} Give balanced advice and recommend professional consultation.",
            QueryType.COMPARISON: f"{base} Compare options objectively, highlighting key differences.",
            QueryType.GENERAL: f"{base} Provide comprehensive, helpful financial guidance."
        }
        
        return prompts.get(query_type, prompts[QueryType.GENERAL])
    
    def _format_history(self) -> str:
        """
        Format recent conversation history.
        """
        if not self.chat_history:
            return "No previous conversation."
        
        # Only include last 2 exchanges to keep prompt manageable
        recent = self.chat_history[-2:]
        return "\n".join([f"Q: {q}\nA: {a}" for q, a in recent])
    
    def get_stats(self) -> Dict:
        """
        Get session statistics.
        """
        return {
            'total_queries': self.stats['total'],
            'query_types': self.stats['types']
        }
        
    def clear_history(self):
        """
        Clear conversation history.
        """
        self.chat_history = []
        
# Main execution functions
async def demo():
    """
    Quick demo of the advanced RAG system.
    """
    print("🚀 Advanced RAG Demo\n")
    
    bot = AdvancedRAGChatbot()
    
    # Test different query types
    queries = [
        "What is compound interest?",
        "How do I calculate ROI?",
        "Should I invest in stocks or bonds?",
        "What's the difference between mutual funds and ETFs?"
    ]
    
    for query in queries:
        print(f"Query: {query}")
        result = await bot.ask(query)
        print(f"Response: {result['response'][:100]}...\n")
        print(f"Type: {result['query_type']}, Confidence: {result['confidence']:.2f}\n")
        print("-" * 50)
        
async def interactive():
    """
    Interactive chat session.
    """
    print("🚀 Advanced RAG Chatbot")
    print("Commands: 'stats, 'clear', 'quit'\n")
    
    bot = AdvancedRAGChatbot()
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'clear':
            bot.clear_history()
            print("History cleared!\n")
            continue
        elif user_input.lower() == 'stats':
            stats = bot.get_stats()
            print(f"stats: {stats}\n")
            continue
        elif not user_input:
            continue
        
        result = await bot.ask(user_input)
        print(f"Bot: {result['response']}\n")
        
if __name__ == "__main__":
    choice = input("Choose mode (1=Demo, 2=Interactive): ")
    
    if choice == "1":
        asyncio.run(demo())
    else:
        asyncio.run(interactive())