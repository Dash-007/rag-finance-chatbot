import os
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Process investopedia documents
def process_csv_docs(docs, content_field='paragraph', keep_metadata=True, metadata_fields=None):
    """
    Process CSV-loaded LangChain documents.
    
    Args:
        docs (list): List of LangChain Document objects.
        content_field (str): The field to use as page_content (default 'paragraph').
        keep_metadata (bool): If True, include all other fields as metadata.
        metadata_fields (list): If provided, only include these fields as metadata
        
    Returns:
        List of processed Document objects
    """
    
    processed_docs = []
    for doc in docs:
        content = doc.page_content.strip()
        if content:
            if keep_metadata:
                if metadata_fields:
                    metadata = {k: v for k, v in doc.metadata.items() if k in metadata_fields}
                else:
                    metadata = doc.metadata.copy()
            else:
                metadata = {}
            processed_docs.append(Document(page_content=content, metadata=metadata))
    return processed_docs

# Load data_samples (finance terms)
loader1 = CSVLoader(file_path='data/data_samples.csv', encoding='utf-8')
doc1 = loader1.load()
print(doc1)
print(type(doc1))
processed_doc1 = process_csv_docs(doc1, content_field='definitions',
                                  keep_metadata=True,
                                  metadata_fields=['terms', 'source'])

# Load investopedia articles
loader2 = CSVLoader(file_path='data/investopedia_articles_001.csv', encoding='utf-8')
doc2 = loader2.load()
processed_doc2 = process_csv_docs(doc2, content_field='paragraph', keep_metadata=True)

loader3 = CSVLoader(file_path='data/investopedia_articles_002.csv', encoding='utf-8')
doc3 = loader3.load()
processed_doc3 = process_csv_docs(doc3, content_field='paragraph', keep_metadata=True)

documents = processed_doc1 + processed_doc2 + processed_doc3

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
print(f"created {len(texts)} chunks")
