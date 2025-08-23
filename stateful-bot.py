import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate

load_dotenv()

class Chatbot:
    """
    Conversational RAG chatbot using ConversationalRetrievalChain.
    """
    def __init__(self, temperature=0.2, model="gpt-4"):
        """
        Initialize the chatbot.
        """
        self.chat_history = []
        
        # Setup embeddings
        self.embeddings = OpenAIEmbeddings(
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
            template="""You are a friendly and knowledgeable financial assistant.
            Have natural conversations while using the provided context to give helpful, accurate responses.
            Be conversational, personable, and engaging. Do not recite information - weave it naturally into your responses.
            Build on our conversation history and ask follow-up questions when appropriate.
            
            Context: {context}
            
            Chat History: {chat_history}
            Human: {question}
            Assistant: 
            """,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Create conversational retrieval chain
        self.qa = ConversationalRetrievalChain.from_ll(
            llm=self.chat,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            combine_docs_chain_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True
        )        