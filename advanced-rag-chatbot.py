import os
import asyncio
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

class QueryType(Enum):
    
