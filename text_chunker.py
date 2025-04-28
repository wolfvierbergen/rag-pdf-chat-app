#!/usr/bin/env python3
# text_chunker.py - Create semantic chunks from document text

"""
This module handles text chunking:
1. Divides document text into semantically meaningful chunks
2. Preserves metadata connection to source document
3. Ensures technical content integrity
4. Supports different chunking strategies

Usage:
  from text_chunker import TextChunker
  from pdf_processor import PDFProcessor
  
  processor = PDFProcessor()
  chunker = TextChunker(chunk_size=500)
  
  document = processor.process("path/to/pdf")
  chunks = chunker.chunk_document(document)
"""

import re
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from pdf_processor import DocumentContent

# Import performance tracking
try:
    from utils.performance_tracker import track_function, performance_tracker
except ImportError:
    print("Warning: Performance tracking not available")
    # Create dummy decorator if tracking is not available
    def track_function(category):
        def decorator(func):
            return func
        return decorator
    
    class DummyTracker:
        def get_peak_memory_mb(self):
            return 0
    
    performance_tracker = DummyTracker()

@dataclass
class TextChunk:
    """Chunk of text with metadata"""
    text: str
    doc_path: str
    doc_title: str
    doc_hash: str
    page_number: int
    section_title: Optional[str]
    chunk_index: int
    token_count: int
    chunk_id: str  # Unique identifier for the chunk
    
    def get_metadata_dict(self) -> Dict[str, Any]:
        """Get metadata as a dictionary"""
        return {
            "doc_path": self.doc_path,
            "doc_title": self.doc_title,
            "doc_hash": self.doc_hash,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "chunk_id": self.chunk_id
        }

class TextChunker:
    """Split text into chunks based on semantic boundaries"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """Initialize the text chunker
        
        Args:
            chunk_size: Target size for chunks in tokens
            overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Initialize tokenizer
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
            self.tokenizer_name = "cl100k_base"
        except:
            # Fallback to simpler tokenizer
            self.tokenizer = SimpleTokenizer()
            self.tokenizer_name = "simple_tokenizer"
    
    def chunk_document(self, document) -> List['TextChunk']:
        """Split document into semantic chunks
        
        Args:
            document: Document to chunk
            
        Returns:
            List of TextChunk objects
        """
        # Track chunk generation time for performance metrics
        start_time = time.time()
        
        # Try different ways to get the text from the document
        if hasattr(document, 'get_full_text') and callable(document.get_full_text):
            text = document.get_full_text()
        elif hasattr(document, 'pages'):
            # Concatenate text from all pages
            text = ' '.join([page.text for page in document.pages if hasattr(page, 'text')])
        elif hasattr(document, 'extracted_text'):
            text = document.extracted_text
        elif hasattr(document, 'content'):
            text = document.content
        else:
            raise AttributeError(f"Document object doesn't have any recognized text attribute. Available attributes: {dir(document)}")
        
        # Create semantic chunks
        chunks = self._create_semantic_chunks(text, document)
        
        # Track generation time for performance
        generation_time = time.time() - start_time
        for chunk in chunks:
            chunk.generation_time = generation_time / len(chunks)
        
        return chunks
    
    def _create_semantic_chunks(self, text: str, document) -> List['TextChunk']:
        """Create chunks based on semantic boundaries
        
        Tries to keep paragraphs, sentences, and sections together when possible.
        Splits on section boundaries first, then paragraphs, then sentences.
        
        Args:
            text: Text to chunk
            document: Source document
            
        Returns:
            List of TextChunk objects
        """
        import re
        
        # Extract metadata from document
        doc_title = getattr(document, 'title', getattr(document, 'filename', 'Unknown'))
        doc_path = getattr(document, 'path', 'Unknown')
        doc_hash = getattr(document, 'hash', getattr(document, 'doc_hash', 'Unknown'))
        
        # Split by sections (marked by headers)
        section_pattern = r'(?:\n\n|\r\n\r\n)(?:[A-Z][^.\n\r]{0,40}[.:]\s*(?:\n|\r\n))'
        sections = re.split(section_pattern, text)
        
        # If sections are too big, split by paragraphs
        paragraphs = []
        for section in sections:
            # Skip empty sections
            if not section.strip():
                continue
                
            # If section is small enough, keep it as is
            section_tokens = self.tokenizer.encode(section)
            if len(section_tokens) <= self.chunk_size:
                paragraphs.append(section)
                continue
                
            # Otherwise split by paragraphs
            section_paragraphs = re.split(r'(?:\n\n|\r\n\r\n)+', section)
            paragraphs.extend(section_paragraphs)
        
        # If paragraphs are too big, split by sentences
        chunks_text = []
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
                
            # Count tokens in paragraph
            paragraph_tokens = self.tokenizer.encode(paragraph)
            paragraph_token_count = len(paragraph_tokens)
            
            # If paragraph fits in current chunk, add it
            if current_tokens + paragraph_token_count <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += paragraph
                current_tokens += paragraph_token_count
                continue
            
            # If paragraph is small enough for its own chunk, finish current chunk and start new one
            if paragraph_token_count <= self.chunk_size:
                if current_chunk:
                    chunks_text.append(current_chunk)
                current_chunk = paragraph
                current_tokens = paragraph_token_count
                continue
            
            # If paragraph is too big, finish current chunk and split paragraph by sentences
            if current_chunk:
                chunks_text.append(current_chunk)
                current_chunk = ""
                current_tokens = 0
            
            # Split paragraph into sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            
            for sentence in sentences:
                # Skip empty sentences
                if not sentence.strip():
                    continue
                    
                # Count tokens in sentence
                sentence_tokens = self.tokenizer.encode(sentence)
                sentence_token_count = len(sentence_tokens)
                
                # If sentence fits in current chunk, add it
                if current_tokens + sentence_token_count <= self.chunk_size:
                    if current_chunk:
                        current_chunk += " "
                    current_chunk += sentence
                    current_tokens += sentence_token_count
                    continue
                
                # If sentence is too big, finish current chunk and split sentence by tokens
                if current_chunk:
                    chunks_text.append(current_chunk)
                
                # If sentence is huge, split it by tokens
                if sentence_token_count > self.chunk_size:
                    sentence_chunks = self._split_by_tokens(sentence_tokens, self.chunk_size, self.overlap)
                    chunks_text.extend([self.tokenizer.decode(chunk) for chunk in sentence_chunks])
                    current_chunk = ""
                    current_tokens = 0
                else:
                    # Start new chunk with this sentence
                    current_chunk = sentence
                    current_tokens = sentence_token_count
        
        # Add final chunk if not empty
        if current_chunk:
            chunks_text.append(current_chunk)
        
        # Create TextChunk objects
        return self._create_text_chunks(chunks_text, doc_title, doc_path, doc_hash)

    def _split_by_tokens(self, tokens, chunk_size, overlap):
        """Split tokens into overlapping chunks"""
        chunks = []
        i = 0
        while i < len(tokens):
            # Add chunk
            end = min(i + chunk_size, len(tokens))
            chunks.append(tokens[i:end])
            
            # Move to next chunk with overlap
            i += chunk_size - overlap
            
            # Avoid creating tiny chunks at the end
            if i + chunk_size > len(tokens) and i < len(tokens) - overlap:
                break
        
        # Add final chunk if needed
        if i < len(tokens):
            chunks.append(tokens[i:])
        
        return chunks
    
    def _create_text_chunks(self, chunks_text, doc_title, doc_path, doc_hash):
        """Create TextChunk objects from text chunks"""
        chunks = []
        
        for i, chunk_text in enumerate(chunks_text):
            # Get token count
            token_count = len(self.tokenizer.encode(chunk_text))
            
            # Create a unique chunk_id based on document hash and chunk index
            chunk_id = f"{doc_hash}_{i}"

            # Create chunk
            chunk = TextChunk(
                text=chunk_text,
                chunk_index=i,
                token_count=token_count,
                doc_title=doc_title,
                doc_path=doc_path,
                doc_hash=doc_hash,
                page_number=0,  # Would need page tracking from PDF processor
                section_title="",  # Would need section tracking from PDF processor
                chunk_id=chunk_id  # Add the chunk_id here
            )
            
            chunks.append(chunk)
        
        return chunks

class SimpleTokenizer:
    """Simple tokenizer fallback using words as tokens"""
    
    def encode(self, text):
        """Split text into word tokens"""
        if not text:
            return []
        return text.split()
    
    def decode(self, tokens):
        """Join tokens into text"""
        return " ".join(tokens)