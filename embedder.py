#!/usr/bin/env python3
# embedder.py - Generate embeddings for text chunks

"""
This module handles embedding generation:
1. Converts text chunks to vector embeddings
2. Manages batching for efficient processing
3. Stores embeddings in FAISS index
4. Preserves connections between embeddings and metadata

Usage:
  from embedder import Embedder
  from text_chunker import TextChunker
  from pdf_processor import PDFProcessor
  
  processor = PDFProcessor()
  chunker = TextChunker()
  embedder = Embedder()
  
  document = processor.process("path/to/pdf")
  chunks = chunker.chunk_document(document)
  embedder.add_chunks(chunks)
  
  # Save index
  embedder.save_index("path/to/index")
"""

import os
import time
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from sentence_transformers import SentenceTransformer
from text_chunker import TextChunk

# Import performance tracking
try:
    from utils.performance_tracker import track_function, performance_tracker, PDFProcessingMetrics
except ImportError:
    print("Warning: Performance tracking not available")
    # Create dummy decorator and classes if tracking is not available
    def track_function(category):
        def decorator(func):
            return func
        return decorator
    
    class DummyTracker:
        def get_peak_memory_mb(self):
            return 0
        
        def record_pdf_processing(self, metrics):
            pass
    
    performance_tracker = DummyTracker()
    
    class PDFProcessingMetrics:
        pass

# Constants
DEFAULT_EMBEDDING_MODEL = "intfloat/e5-base-v2"

class Embedder:
    """Generate and store embeddings for text chunks"""
    
    def __init__(self, 
                 model_name: str = DEFAULT_EMBEDDING_MODEL, 
                 batch_size: int = 32,
                 index_folder: str = "processed/index"):
        """Initialize the embedder
        
        Args:
            model_name: Name of the embedding model to use
            batch_size: Number of chunks to process at once
            index_folder: Folder to store the index
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.index_folder = index_folder
        
        # Initialize model
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            print(f"Initialized embedding model: {model_name} with dimension {self.embedding_dimension}")
        except Exception as e:
            print(f"Error initializing embedding model: {e}")
            raise
        
        # Initialize index, corpus and metadata
        self.index = None
        self.metadata = []
        self.corpus = []
    
    def _prepare_text_for_embedding(self, chunk: TextChunk) -> str:
        """Format text chunk for embedding model"""
        # Format as 'passage: text' for better performance with models like E5
        return f"passage: {chunk.text}"
    
    @track_function("Embedding Generation")
    def generate_embeddings(self, chunks: List[TextChunk]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Generate embeddings for text chunks
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            Tuple of (embeddings array, metadata list)
        """
        if not chunks:
            return np.array([]), []
        
        # Prepare text for embedding
        texts = [self._prepare_text_for_embedding(chunk) for chunk in chunks]
        chunk_metadata = [chunk.get_metadata_dict() for chunk in chunks]
        
        # Generate embeddings in batches
        all_embeddings = []
        
        start_time = time.time()
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            batch_chunks = chunks[i:i+self.batch_size]
            
            # Generate embeddings for batch
            batch_embeddings = self.model.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=(len(batch_texts) > 10)
            )
            
            # Attach embeddings to chunk objects
            for j, embedding in enumerate(batch_embeddings):
                # Add the embedding attribute to the chunk object
                setattr(batch_chunks[j], 'embedding', embedding)
            
            all_embeddings.append(batch_embeddings)
        
        # Combine batches
        embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
        
        embedding_time = time.time() - start_time
        print(f"Generated {len(chunks)} embeddings in {embedding_time:.2f} seconds")
        
        return embeddings, chunk_metadata
    
    @track_function("FAISS Index Update")
    def add_chunks(self, chunks: List[TextChunk], record_performance: bool = True) -> None:
        """Add chunks to the index
        
        Args:
            chunks: List of TextChunk objects
            record_performance: Whether to record performance metrics
        """
        if not chunks:
            return
        
        # Group chunks by document for performance metrics
        doc_chunks = {}
        for chunk in chunks:
            if chunk.doc_hash not in doc_chunks:
                doc_chunks[chunk.doc_hash] = {
                    'path': chunk.doc_path,
                    'chunks': []
                }
            doc_chunks[chunk.doc_hash]['chunks'].append(chunk)
        
        # Process each document
        for doc_hash, doc_info in doc_chunks.items():
            doc_chunks_list = doc_info['chunks']
            doc_path = doc_info['path']
            
            # Generate embeddings for this document's chunks
            start_time = time.time()
            embeddings, chunk_metadata = self.generate_embeddings(doc_chunks_list)
            embedding_time = time.time() - start_time
            
            if len(embeddings) == 0:
                print(f"Warning: No embeddings generated for {doc_path}")
                continue
            
            # Initialize index if needed
            if self.index is None:
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
            
            # Update index
            self.index.add(embeddings.astype(np.float32))
            
            # Update metadata and corpus
            for i, chunk in enumerate(doc_chunks_list):
                self.metadata.append(chunk_metadata[i])
                self.corpus.append(chunk.text)
            
            # Record performance if requested
            if record_performance:
                num_tokens = sum(chunk.token_count for chunk in doc_chunks_list)
                total_text_length = sum(len(chunk.text) for chunk in doc_chunks_list)
                
                # Get document size
                try:
                    doc_size = os.path.getsize(doc_path)
                except (OSError, FileNotFoundError):
                    doc_size = 0
                
                # Calculate processing time based on chunk generation time if available
                chunk_processing_time = 0.0
                if hasattr(doc_chunks_list[0], 'generation_time'):
                    chunk_processing_time = sum(chunk.generation_time for chunk in doc_chunks_list 
                                              if hasattr(chunk, 'generation_time'))
                
                # Calculate tokens per second if processing time is available
                tokens_per_second = 0.0
                if chunk_processing_time > 0:
                    tokens_per_second = num_tokens / chunk_processing_time
                
                # Record metrics
                try:
                    metrics = PDFProcessingMetrics(
                        pdf_path=doc_path,
                        pdf_size_bytes=doc_size,
                        extracted_text_length=total_text_length,
                        num_tokens=num_tokens,
                        num_chunks=len(doc_chunks_list),
                        processing_time_seconds=chunk_processing_time,  # Use actual time if available
                        tokens_per_second=tokens_per_second,  # Calculate if possible
                        embedding_time_seconds=embedding_time,
                        peak_memory_mb=performance_tracker.get_peak_memory_mb(),
                        timestamp=datetime.now().isoformat(),
                        chunk_size=doc_chunks_list[0].token_count if doc_chunks_list else 0,
                        tokenizer_name="o200k_base",  # This should be passed from chunker
                        embedding_model=self.model_name
                    )
                    performance_tracker.record_pdf_processing(metrics)
                except:
                    print("Warning: Could not record performance metrics")
    
    def save_index(self, index_path: Optional[str] = None) -> None:
        """Save the index, corpus and metadata
        
        Args:
            index_path: Directory to save the index (default: self.index_folder)
        """
        if self.index is None or len(self.metadata) == 0:
            print("Warning: No index to save")
            return
        
        if index_path is None:
            index_path = self.index_folder
        
        # Create directory if it doesn't exist
        os.makedirs(index_path, exist_ok=True)
        
        # Save index
        index_file = os.path.join(index_path, "faiss_index.idx")
        metadata_file = os.path.join(index_path, "metadata.pkl")
        corpus_file = os.path.join(index_path, "corpus.pkl")
        
        faiss.write_index(self.index, index_file)
        
        with open(metadata_file, "wb") as f:
            pickle.dump(self.metadata, f)
        
        with open(corpus_file, "wb") as f:
            pickle.dump(self.corpus, f)
        
        print(f"Saved index with {self.index.ntotal} vectors")
    
    def load_index(self, index_path: Optional[str] = None) -> bool:
        """Load index, corpus and metadata
        
        Args:
            index_path: Directory to load the index from (default: self.index_folder)
            
        Returns:
            Success status
        """
        if index_path is None:
            index_path = self.index_folder
        
        # Check if files exist
        index_file = os.path.join(index_path, "faiss_index.idx")
        metadata_file = os.path.join(index_path, "metadata.pkl")
        corpus_file = os.path.join(index_path, "corpus.pkl")
        
        if not all(os.path.exists(f) for f in [index_file, metadata_file, corpus_file]):
            print("Warning: Index files not found")
            return False
        
        # Load index
        try:
            self.index = faiss.read_index(index_file)
            
            with open(metadata_file, "rb") as f:
                self.metadata = pickle.load(f)
            
            with open(corpus_file, "rb") as f:
                self.corpus = pickle.load(f)
            
            print(f"Loaded index with {self.index.ntotal} vectors")
            return True
        
        except Exception as e:
            print(f"Error loading index: {e}")
            return False

# Example usage
if __name__ == "__main__":
    import sys
    from pdf_processor import PDFProcessor
    from text_chunker import TextChunker
    
    if len(sys.argv) != 2:
        print("Usage: python embedder.py path/to/pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    try:
        # Process PDF
        processor = PDFProcessor()
        chunker = TextChunker(chunk_size=500)
        embedder = Embedder()
        
        # Process document
        document = processor.process(pdf_path)
        chunks = chunker.chunk_document(document)
        
        print(f"Adding {len(chunks)} chunks to index...")
        embedder.add_chunks(chunks)
        
        # Save index
        embedder.save_index()
        
        print(f"Processed document: {document.filename}")
        print(f"Total chunks: {len(chunks)}")
        print(f"Total vectors: {embedder.index.ntotal}")
    
    except Exception as e:
        print(f"Error: {e}")