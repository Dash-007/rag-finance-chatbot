import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import mimetypes

# Document processing libraries
import PyPDF2
import pdfplumber
from docx import Document as DocxDocument
import pandas as pd
from bs4 import BeautifulSoup
import markdown

# OCR and image processing
try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    
# NLP libraries for content analysis
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY: False
    
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """
    Enhanced metadata for enterprise documents.
    """
    file_path: str
    file_name: str
    file_size: int
    file_type: str
    mime_type: str
    document_hash: str
    processed_date: datetime
    source_system: Optional[str]
    department: Optional[str] = None
    classification: str = "public" # public, internal, confidential, restricted
    author: Optional[str] = None
    creation_date: Optional[datetime] = None
    language: str = "en"
    document_type: Optional[str] = None # financial_report, legal_contract
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class ProcessingResult:
    """
    Result of document processing.
    """
    success: bool
    Document: List[Document]
    metadata: DocumentMetadata
    error: Optional[str] = None
    processing_time: float = 0.0
    extracted_elements: Dict[str, Any] = field(default_factory=dict)
    
class BaseDocumentProcessor(ABC):
    """
    Abstract base class for document processors.
    """
    
    @abstractmethod
    async def process(self, file_path: Path, metadata: Dict[str, Any]) -> ProcessingResult:
        """Process a document and return structured content."""
        pass
    
    @abstractmethod
    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle the file type."""
        pass
    
    def _create_metadata(self, file_path: Path, **kwargs) -> DocumentMetadata:
        """Create metadata for a document."""
        file_stat = file_path.stat()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        # Generate file hash for deduplication
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
            
        return DocumentMetadata(
            file_path=str(file_path),
            file_name=file_path.name,
            file_size=file_stat.st_size,
            file_type=file_path.suffix.lower(),
            mime_type=mime_type or "unknown",
            document_hash=file_hash,
            processed_date=datetime.now(),
            **kwargs
        )