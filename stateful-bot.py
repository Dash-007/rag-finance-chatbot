import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate

load_dotenv()

class Chatbot:
    """
    Conversational RAG chatbot using ConversationalRetrievalChain.
    """
    def __init__(self, temperature=0.0, model="gpt-4"):
        """
        Initialize the chatbot.
        """
        self.chat_history = []
        
        # Setup embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=512,
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Setup vector store
        self.vectorstore = PineconeVectorStore(
            index_name=os.environ.get("INDEX_NAME"),
            embedding=self.embeddings
        )
        
        # Setup chat model
        self.chat = ChatOpenAI(
            temperature=temperature,
            model=model,
            openai_api_key=os.environ.get("OPENAI_API_KEY")            
        )
        
        self.qa_prompt = PromptTemplate(
            template="""
            You are a friendly and knowledgeable financial assistant. Have natural conversations while using the provided context to give helpful, accurate responses. Be conversational, personable, and engaging. Do not recite information - weave it naturally into your responses. Build on our conversation history and ask follow-up questions when appropriate.
            Context: {context}
            Question: {question} 
            """,
            input_variables=["context", "question"]
        )
        
        self.condense_question_prompt = PromptTemplate(
            template="""
            Given the following conversation and a follow up question...
            """,
            input_variables=["chat_history", "question"]
        )
        
        # Create conversational retrieval chain
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.chat,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            condense_question_prompt=self.condense_question_prompt,
            combine_docs_chain_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True
        )
        
    def ask(self, question):
        """
        Ask a question and get a conversational response.
        """
        try:
            result = self.qa.invoke({
                "question": question,
                "chat_history": self.chat_history
            })
            
            # Update chat history
            self.chat_history.append((question, result["answer"]))
            
            return {
                "response": result["answer"],
                "sources": self._extract_source_info(result.get("source_documents", []))
            }
            
        except Exception as e:
            print(f"Debug - Error details: {str(e)}")
            return {
                "response": "Sorry, I encountered an issue. Could you try asking that again?",
                "sources": [],
                "error": str(e)
            }
            
    def _extract_source_info(self, docs):
        """
        Extract relevant source information (terms and title only).
        """
        sources = []
        
        for doc in docs:
            source_info = {}
            metadata = doc.metadata
            
            if 'terms' in metadata and metadata['terms']:
                source_info['terms'] = metadata['terms']
            if 'title' in metadata and metadata['title']:
                source_info['title'] = metadata['title']
                
            if source_info:
                sources.append(source_info)
                
        return sources
    
    def clear_history(self):
        """
        Clear conversation history.
        """
        self.chat_history = []
        
    def get_history(self):
        """
        Get conversation history
        """
        return self.chat_history
    
def main():
    """
    Interactuve chatbot session.
    """
    print("🤖 Financial Chatbot - Let's talk about your finances!")
    print("Commands: 'history' to view conversation, 'clear' to reset, 'quit' to exit\n")
    
    bot = Chatbot()
    
    while True:
        user_input  = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            print("take care! 👋")
            break
        elif user_input.lower() == 'clear':
            bot.clear_history()
            print("🧹 Chat history cleared!\n")
            continue
        elif user_input.lower() == 'history':
            history = bot.get_history()
            if history:
                print("\n💬 Our conversation: ")
                for i, (q, a) in enumerate(history, 1):
                    print(f"{i}.    You: {q[:60]}{'...' if len(q) > 60 else ''}")
                    print(f"        Bot: {a[:60]}{'...' if len(a) > 60 else ''}")
            else:
                print("No conversation yet!")
            print()
            continue
        elif not user_input:
            continue
        
        # Get response
        result = bot.ask(user_input)
        print(f"\n🤖 Chatbot: {result['response']}")
        
        # Show sources briefly if relevant
        if result['sources']:
            print(f"\n💡 Drew from: {','.join([s.get('term', s.get('title', 'financial knowledge')) for s in result['sources'][:2]])}")
            
        print()
        
if __name__ == "__main__":
    main()