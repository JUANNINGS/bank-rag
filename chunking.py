"""
Text Chunking Module
Handles document splitting and preprocessing for RAG pipeline
Following patterns from RAG.ipynb
"""

import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import get_rag_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BankDocumentChunker:
    """
    Document chunker for banking documents
    Uses RecursiveCharacterTextSplitter for intelligent splitting
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize document chunker
        
        Args:
            chunk_size: Size of each chunk in characters (default from config)
            chunk_overlap: Overlap between chunks (default from config)
        """
        config = get_rag_config()
        
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap
        
        # Initialize text splitter
        # RecursiveCharacterTextSplitter tries to split on:
        # 1. Double newlines (paragraphs)
        # 2. Single newlines
        # 3. Spaces
        # 4. Characters (as last resort)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info(
            f"Initialized chunker with chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of chunked Document objects with metadata
        """
        if not documents:
            logger.warning("No documents provided to split")
            return []
        
        # Split all documents
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            # Preserve original metadata
            original_source = chunk.metadata.get('source', 'unknown')
            
            # Add chunk metadata
            chunk.metadata.update({
                'chunk_id': f"chunk_{i}",
                'chunk_index': i,
                'chunk_size': len(chunk.page_content),
                'original_source': original_source
            })
        
        logger.info(
            f"Split {len(documents)} document(s) into {len(chunks)} chunk(s)"
        )
        logger.info(
            f"Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks) if chunks else 0} characters"
        )
        
        return chunks
    
    def split_text(self, texts: List[str]) -> List[Document]:
        """
        Split raw text strings into chunks
        
        Args:
            texts: List of text strings
            
        Returns:
            List of chunked Document objects
        """
        chunks = self.text_splitter.create_documents(texts)
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata = {
                'chunk_id': f"chunk_{i}",
                'chunk_index': i,
                'chunk_size': len(chunk.page_content),
                'source': 'raw_text'
            }
        
        logger.info(f"Split {len(texts)} text(s) into {len(chunks)} chunk(s)")
        return chunks
    
    def get_chunk_statistics(self, chunks: List[Document]) -> dict:
        """
        Get statistics about chunks
        
        Args:
            chunks: List of chunked documents
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'total_characters': 0
            }
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) // len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes),
            'sources': len(set(c.metadata.get('source', 'unknown') for c in chunks))
        }


def chunk_banking_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[Document]:
    """
    Convenience function to chunk banking documents
    
    Args:
        documents: List of Document objects
        chunk_size: Optional custom chunk size
        chunk_overlap: Optional custom overlap
        
    Returns:
        List of chunked Document objects
    """
    chunker = BankDocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = chunker.split_documents(documents)
    
    # Print statistics
    stats = chunker.get_chunk_statistics(chunks)
    logger.info(f"Chunk Statistics:")
    logger.info(f"  Total chunks: {stats['total_chunks']}")
    logger.info(f"  Avg chunk size: {stats['avg_chunk_size']} characters")
    logger.info(f"  Size range: {stats['min_chunk_size']} - {stats['max_chunk_size']} characters")
    logger.info(f"  From {stats['sources']} different source(s)")
    
    return chunks


# Example usage
if __name__ == "__main__":
    from document_loader import load_banking_documents
    
    print("="*60)
    print("Testing Bank Document Chunker")
    print("="*60)
    
    try:
        # Load documents
        print("\n1. Loading documents...")
        documents = load_banking_documents("./documents")
        print(f"✓ Loaded {len(documents)} document(s)")
        
        # Chunk documents
        print("\n2. Chunking documents...")
        chunks = chunk_banking_documents(documents)
        print(f"✓ Created {len(chunks)} chunk(s)")
        
        # Show sample chunk
        if chunks:
            print(f"\n3. Sample chunk:")
            print(f"   Chunk ID: {chunks[0].metadata.get('chunk_id')}")
            print(f"   Source: {chunks[0].metadata.get('original_source')}")
            print(f"   Size: {chunks[0].metadata.get('chunk_size')} characters")
            print(f"   Content preview:\n   {chunks[0].page_content[:200]}...")
            
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
