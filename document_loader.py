"""
Document Loader Module
Handles loading and parsing of various document formats (PDF, DOCX, HTML, TXT)
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    DirectoryLoader
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BankDocumentLoader:
    """
    Unified document loader for banking documents
    Supports: PDF, DOCX, HTML, TXT
    """
    
    SUPPORTED_EXTENSIONS = {
        '.txt': TextLoader,
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.html': UnstructuredHTMLLoader,
        '.htm': UnstructuredHTMLLoader
    }
    
    def __init__(self, documents_path: str = "./documents"):
        """
        Initialize document loader
        
        Args:
            documents_path: Path to documents directory
        """
        self.documents_path = Path(documents_path)
        if not self.documents_path.exists():
            raise ValueError(f"Documents path does not exist: {documents_path}")
        
        logger.info(f"Initialized BankDocumentLoader with path: {documents_path}")
    
    def load_single_document(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load a single document file
        
        Args:
            file_path: Path to document file
            
        Returns:
            List of Document objects (may contain multiple pages/sections)
            
        Raises:
            ValueError: If file type not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file extension
        extension = file_path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )
        
        # Load document using appropriate loader
        loader_class = self.SUPPORTED_EXTENSIONS[extension]
        loader = loader_class(str(file_path))
        
        try:
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path.name),
                    'file_path': str(file_path),
                    'file_type': extension,
                    'document_type': 'banking_document'
                })
            
            logger.info(f"Loaded {len(documents)} page(s) from {file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {str(e)}")
            raise
    
    def load_all_documents(
        self, 
        recursive: bool = False,
        glob_pattern: Optional[str] = None
    ) -> List[Document]:
        """
        Load all supported documents from documents directory
        
        Args:
            recursive: Whether to search subdirectories
            glob_pattern: Optional glob pattern to filter files (e.g., "*.pdf")
            
        Returns:
            List of all loaded Document objects
        """
        all_documents = []
        
        # Determine which files to load
        if glob_pattern:
            if recursive:
                file_paths = list(self.documents_path.rglob(glob_pattern))
            else:
                file_paths = list(self.documents_path.glob(glob_pattern))
        else:
            # Load all supported file types
            file_paths = []
            for ext in self.SUPPORTED_EXTENSIONS.keys():
                if recursive:
                    file_paths.extend(self.documents_path.rglob(f"*{ext}"))
                else:
                    file_paths.extend(self.documents_path.glob(f"*{ext}"))
        
        logger.info(f"Found {len(file_paths)} document(s) to load")
        
        # Load each document
        for file_path in file_paths:
            try:
                documents = self.load_single_document(file_path)
                all_documents.extend(documents)
            except Exception as e:
                logger.warning(f"Skipping {file_path.name}: {str(e)}")
                continue
        
        logger.info(f"Successfully loaded {len(all_documents)} document(s) total")
        return all_documents
    
    def load_documents_by_type(self, file_types: List[str]) -> List[Document]:
        """
        Load documents of specific types only
        
        Args:
            file_types: List of extensions to load (e.g., ['.pdf', '.txt'])
            
        Returns:
            List of loaded Document objects
        """
        all_documents = []
        
        for file_type in file_types:
            if file_type not in self.SUPPORTED_EXTENSIONS:
                logger.warning(f"Unsupported file type: {file_type}, skipping")
                continue
            
            file_paths = list(self.documents_path.glob(f"*{file_type}"))
            logger.info(f"Found {len(file_paths)} {file_type} file(s)")
            
            for file_path in file_paths:
                try:
                    documents = self.load_single_document(file_path)
                    all_documents.extend(documents)
                except Exception as e:
                    logger.warning(f"Skipping {file_path.name}: {str(e)}")
                    continue
        
        return all_documents
    
    def get_document_summary(self, documents: List[Document]) -> dict:
        """
        Get summary statistics about loaded documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with summary statistics
        """
        if not documents:
            return {
                'total_documents': 0,
                'total_characters': 0,
                'file_types': {},
                'sources': []
            }
        
        # Calculate statistics
        total_chars = sum(len(doc.page_content) for doc in documents)
        file_types = {}
        sources = set()
        
        for doc in documents:
            file_type = doc.metadata.get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
            sources.add(doc.metadata.get('source', 'unknown'))
        
        return {
            'total_documents': len(documents),
            'total_characters': total_chars,
            'average_length': total_chars // len(documents) if documents else 0,
            'file_types': file_types,
            'sources': sorted(list(sources)),
            'unique_sources': len(sources)
        }


def load_banking_documents(
    documents_path: str = "./documents",
    file_types: Optional[List[str]] = None
) -> List[Document]:
    """
    Convenience function to load banking documents
    
    Args:
        documents_path: Path to documents directory
        file_types: Optional list of file types to load
        
    Returns:
        List of Document objects
    """
    loader = BankDocumentLoader(documents_path)
    
    if file_types:
        documents = loader.load_documents_by_type(file_types)
    else:
        documents = loader.load_all_documents()
    
    # Print summary
    summary = loader.get_document_summary(documents)
    logger.info(f"Loaded {summary['total_documents']} documents from {summary['unique_sources']} files")
    logger.info(f"Total characters: {summary['total_characters']:,}")
    logger.info(f"File types: {summary['file_types']}")
    
    return documents


# Example usage
if __name__ == "__main__":
    # Test the document loader
    print("="*60)
    print("Testing Bank Document Loader")
    print("="*60)
    
    try:
        # Load all documents
        documents = load_banking_documents("./documents")
        
        print(f"\n✓ Successfully loaded {len(documents)} document(s)")
        
        # Show first document preview
        if documents:
            print(f"\nFirst document preview:")
            print(f"Source: {documents[0].metadata.get('source')}")
            print(f"Content preview: {documents[0].page_content[:200]}...")
            
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
