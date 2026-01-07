# context.py
from typing import Optional, List, Dict, Any, Union
import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json
import hashlib
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentMetadata:
    """Class to handle document metadata consistently."""
    def __init__(self, 
                 source: str,
                 doc_type: str = "pdf",
                 author: Optional[str] = None,
                 created_at: Optional[datetime] = None,
                 tags: Optional[List[str]] = None,
                 **kwargs):
        self.source = source
        self.doc_type = doc_type
        self.author = author
        self.created_at = created_at or datetime.now()
        self.tags = tags or []
        self.additional_metadata = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        return {
            "source": self.source,
            "doc_type": self.doc_type,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
            **self.additional_metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentMetadata':
        """Create metadata instance from dictionary."""
        created_at = datetime.fromisoformat(data.pop("created_at")) if "created_at" in data else None
        tags = data.pop("tags", [])
        source = data.pop("source")
        doc_type = data.pop("doc_type", "pdf")
        author = data.pop("author", None)
        
        return cls(
            source=source,
            doc_type=doc_type,
            author=author,
            created_at=created_at,
            tags=tags,
            **data
        )

class ContextManager:
    """
    Manages RAG-based context using ChromaDB for vector storage.
    
    This class handles:
    - Document indexing and chunking
    - Vector storage management
    - Context retrieval
    - State persistence
    """
    
    def __init__(self, 
                 collection_name: str, 
                 persist_dir: str = "context_db",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the context manager.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_dir: Directory to persist the vector database
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between consecutive chunks
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Ensure persist directory exists
        os.makedirs(persist_dir, exist_ok=True)
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(path=persist_dir)
            logger.info(f"Initialized ChromaDB client with persist_dir: {persist_dir}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {str(e)}")
            raise
        
        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Using cosine similarity
            )
            logger.info(f"Initialized collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error creating/getting collection: {str(e)}")
            raise
        
        # Initialize text splitter
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # State variables
        self._current_query: Optional[str] = None
        self._current_context: Optional[str] = None
        self._indexed_documents: Dict[str, DocumentMetadata] = {}
    
    @classmethod
    def initialize(cls, 
                  collection_name: Optional[str] = None, 
                  persist_dir: str = "context_db",
                  **kwargs) -> 'ContextManager':
        """
        Factory method to create and initialize a new ContextManager.
        
        Args:
            collection_name: Optional name for the ChromaDB collection
            persist_dir: Directory to persist the vector database
            **kwargs: Additional arguments for ContextManager initialization
            
        Returns:
            ContextManager: Initialized context manager instance
        """
        if not collection_name:
            # Generate a unique collection name if none provided
            random_suffix = hashlib.md5(os.urandom(8)).hexdigest()[:8]
            collection_name = f"collection_{random_suffix}"
        
        return cls(collection_name=collection_name, persist_dir=persist_dir, **kwargs)
    
    def _generate_document_id(self, content: str, metadata: Optional[Dict] = None) -> str:
        """
        Generate a unique ID for a document chunk.
        
        Args:
            content: The content to generate an ID for
            metadata: Optional metadata to include in ID generation
            
        Returns:
            str: Unique identifier for the chunk
        """
        # Combine content with metadata for ID generation if provided
        id_content = content
        if metadata:
            id_content += json.dumps(metadata, sort_keys=True)
        return hashlib.sha256(id_content.encode()).hexdigest()[:16]
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            str: Extracted text content
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF is empty or unreadable
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                if len(pdf_reader.pages) == 0:
                    raise ValueError(f"PDF file is empty: {pdf_path}")
                
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
                if not text.strip():
                    raise ValueError(f"No text content extracted from PDF: {pdf_path}")
                
                return text
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            raise
    
    def index_document(self, 
                      pdf_path: str, 
                      metadata: Optional[Union[Dict[str, Any], DocumentMetadata]] = None) -> bool:
        """
        Index a PDF document into the vector store.
        
        Args:
            pdf_path: Path to the PDF file
            metadata: Optional metadata about the document
            
        Returns:
            bool: True if indexing was successful
            
        Raises:
            ValueError: If document is empty or invalid
        """
        try:
            # Convert metadata to DocumentMetadata if it's a dict
            if isinstance(metadata, dict):
                metadata = DocumentMetadata.from_dict(metadata)
            elif metadata is None:
                metadata = DocumentMetadata(source=os.path.basename(pdf_path))
            
            # Extract text from PDF
            text = self._extract_text_from_pdf(pdf_path)
            
            # Split text into chunks
            chunks = self._text_splitter.split_text(text)
            if not chunks:
                raise ValueError(f"No valid chunks generated from document: {pdf_path}")
            
            # Prepare documents for insertion
            ids = [self._generate_document_id(chunk, metadata.to_dict()) for chunk in chunks]
            metadatas = [metadata.to_dict() for _ in chunks]
            
            # Add to ChromaDB
            self.collection.add(
                documents=chunks,
                ids=ids,
                metadatas=metadatas
            )
            
            # Store document metadata
            self._indexed_documents[pdf_path] = metadata
            logger.info(f"Successfully indexed document: {pdf_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error indexing document {pdf_path}: {str(e)}")
            return False

    def set_query(self, 
                query: str, 
                num_results: int = 3,
                filter_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Set and execute a new query for context retrieval.
        
        Args:
            query: The query to search for context
            num_results: Number of relevant chunks to retrieve
            filter_metadata: Optional metadata filters for the query
            
        Returns:
            str: Retrieved context as a formatted string
        """
        return self.query(query, num_results, filter_metadata)
    
    def query(self, 
             query: str, 
             num_results: int = 3,
             filter_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Query the context and return relevant information.
        
        Args:
            query: The query to search for context
            num_results: Number of relevant chunks to retrieve
            filter_metadata: Optional metadata filters for the query
            
        Returns:
            str: Retrieved context as a formatted string
        """
        try:
            self._current_query = query
            
            # Prepare query parameters
            query_params = {
                "query_texts": [query],
                "n_results": num_results
            }
            if filter_metadata:
                query_params["where"] = filter_metadata
            
            # Execute query
            results = self.collection.query(**query_params)
            
            if not results['documents']:
                self._current_context = ""
                return ""
            
            # Format the context with metadata
            context_parts = []
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
                source = metadata.get('source', 'Unknown source')
                context_parts.append(
                    f"Relevant Context {i} (from {source}):\n{doc}\n"
                )
            
            self._current_context = "\n".join(context_parts)
            return self._current_context
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            self._current_context = ""
            return ""
    
    def get_document_metadata(self, source: str) -> Optional[DocumentMetadata]:
        """
        Retrieve metadata for a specific document.
        
        Args:
            source: Source identifier of the document
            
        Returns:
            Optional[DocumentMetadata]: Document metadata if found
        """
        return self._indexed_documents.get(source)
    
    def list_indexed_documents(self) -> List[Dict[str, Any]]:
        """
        Get a list of all indexed documents and their metadata.
        
        Returns:
            List[Dict[str, Any]]: List of document metadata
        """
        return [
            {
                "source": source,
                "metadata": metadata.to_dict()
            }
            for source, metadata in self._indexed_documents.items()
        ]
    
    def clear_index(self) -> bool:
        """
        Clear all indexed documents from the collection.
        
        Returns:
            bool: True if successful
        """
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self._indexed_documents.clear()
            self._current_context = None
            self._current_query = None
            logger.info(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing index: {str(e)}")
            return False
    
    @property
    def current_query(self) -> Optional[str]:
        """Get the current context query."""
        return self._current_query

    @property
    def response(self) -> Optional[str]:
        """Get the current retrieved context response."""
        return self._current_context
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save the current state of the context manager.
        
        Returns:
            Dict[str, Any]: Current state
        """
        return {
            "current_query": self._current_query,
            "current_context": self._current_context,
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "indexed_documents": {
                source: metadata.to_dict()
                for source, metadata in self._indexed_documents.items()
            }
        }
    
    def load_state(self, state: Dict[str, Any]):
        """
        Load a saved state.
        
        Args:
            state: State dictionary to load
        """
        self._current_query = state.get("current_query")
        self._current_context = state.get("current_context")
        self.chunk_size = state.get("chunk_size", self.chunk_size)
        self.chunk_overlap = state.get("chunk_overlap", self.chunk_overlap)
        
        # Restore indexed documents
        self._indexed_documents = {
            source: DocumentMetadata.from_dict(metadata_dict)
            for source, metadata_dict in state.get("indexed_documents", {}).items()
        }
        
        logger.info("Successfully loaded context manager state")