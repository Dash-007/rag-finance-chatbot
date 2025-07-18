import os
from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader
from langchain.docstore.document import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Load data_samples (finance terms)
loader1 = CSVLoader(file_path='data/data_samples.csv')
doc1 = loader1.load()

# Load investopedia articles
loader2 = CSVLoader(file_path='data/investopedia_articles.csv')
doc2 = loader2.load()

# Process investopedia documents
doc2_processed =[]
for doc in doc2:
    row = doc.page_content
    content = row.get('paragraph', '')
    if content.strip(): # Only include non-empty paragraphs
        metadata = {k: v for k, v in row.items() if k != 'paragraph'}
        doc2_processed.append(Document(page_content=content, metadata=metadata))

documents = doc1 + doc2_processed