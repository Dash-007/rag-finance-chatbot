import os
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Process investopedia documents
def process_csv_docs(docs, keep_metadata=True, metadata_fields=None):
    """
    Process CSV-loaded LangChain documents.
    
    Args:
        docs (list): List of LangChain Document objects.
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

# Load CSV files with Error Handling
def safe_load_csv(file_path, encoding='utf-8', source_column=None):
    """
    Load CSV file into LangChain documents with error handling.
    
    Args:
        file_path (str): Path to CSV file
        encoding (str): File encoding, defaults to 'utf-8' 
        source_column (str): Column to use as document content, optional
    
    Returns:
        list: LangChain Document objects, empty list if loading fails
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")       
        loader_kwargs = {'file_path': file_path, 'encoding': encoding}
      
        if source_column:
            loader_kwargs['source_column'] = source_column
            
        loader = CSVLoader(**loader_kwargs)
        docs = loader.load()
        
        if not docs:
            print(f"Warning: No documents loaded from {file_path}")
            
        return docs
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return []

# Load data_samples (finance terms)
doc1 = safe_load_csv('data/data_samples.csv', source_column='definitions')
print(f"Loaded {len(doc1)} documents from data_samples.csv")
processed_doc1 = process_csv_docs(doc1,
                                  keep_metadata=True,
                                  metadata_fields=['terms', 'source'])

# Load investopedia articles
doc2 = safe_load_csv('data/investopedia_articles_001.csv', source_column='paragraph')
print(f"Loaded {len(doc2)} documents from investopedia_articles_001.csv")
processed_doc2 = process_csv_docs(doc2, keep_metadata=True, metadata_fields=['title'])

doc3 = safe_load_csv('data/investopedia_articles_002.csv', source_column='paragraph')
print(f"Loaded {len(doc3)} documents from investopedia_articles_002.csv")
processed_doc3 = process_csv_docs(doc3, keep_metadata=True, metadata_fields=['title'])

documents = processed_doc1 + processed_doc2 + processed_doc3
print(f"Total documents before chunking: {len(documents)}")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
print(f"created {len(texts)} chunks")

# Create embeddings
print("Creating embeddings and uploading to Pinecone...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512, openai_api_key=os.environ.get("OPENAI_API_KEY"))

# Create vector store
vector_store = PineconeVectorStore.from_documents(documents=texts, embedding=embeddings, index_name=os.environ.get("INDEX_NAME"))

print(f"Successfully uploaded {len(texts)} chunks to Pinecone index: {os.environ.get('INDEX_NAME')}")
print("RAG ingestion complete!")