#!/usr/bin/env python3
# retriever.py - Retrieve relevant chunks and generate responses

"""
This module handles retrieval and response generation:
1. Performs vector similarity search to find relevant chunks
2. Combines chunks into a coherent context
3. Uses the LLM to generate responses with citations
4. Integrates with the performance tracking system

Usage:
  from retriever import Retriever, ResponseResult
  from query_processor import QueryProcessor
  
  retriever = Retriever()
  processor = QueryProcessor()
  
  query = "What is the impact of transformer architecture on NLP?"
  processed_query = processor.process_query(query)
  response = retriever.query(processed_query)
  
  print(response.answer)
"""

import os
import time
import requests
import pickle
import faiss
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from sentence_transformers import SentenceTransformer
from query_processor import QueryProcessor, ProcessedQuery

# Import performance tracking
try:
    from utils.performance_tracker import track_function, performance_tracker, QueryMetrics
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
        
        def record_query(self, metrics):
            pass
    
    performance_tracker = DummyTracker()
    
    class QueryMetrics:
        pass

# Constants
DEFAULT_EMBEDDING_MODEL = "intfloat/e5-base-v2"
DEFAULT_OLLAMA_MODEL = "llama3:8b"
OLLAMA_API_BASE = "http://localhost:11434/api"
DEFAULT_TOP_K = 3
DEFAULT_TEMPERATURE = 0.6
DEFAULT_MAX_TOKENS = 1000

@dataclass
class RetrievedChunk:
    """Retrieved chunk with metadata and relevance score"""
    text: str
    metadata: Dict[str, Any]
    distance: float
    
    def get_citation(self) -> str:
        """Get a citation string for this chunk"""
        title = self.metadata.get('doc_title', 'Unknown')
        page = self.metadata.get('page_number', 0)
        section = self.metadata.get('section_title', 'Unknown Section')
        
        return f"[{title}, Page {page}{', ' + section if section else ''}]"

@dataclass
class RetrievalResult:
    """Results of a retrieval operation"""
    query: str
    chunks: List[RetrievedChunk]
    retrieval_time: float


@dataclass
class ResponseResult:
    """Response from the LLM"""
    query: str
    answer: str
    response_time: float
    retrieval_time: float  # Add this line
    chunks_used: List[RetrievedChunk]
    model_name: str

class Retriever:
    """Retrieve relevant chunks and generate responses"""
    
    def __init__(self, 
                 model_name: str = DEFAULT_EMBEDDING_MODEL,
                 index_folder: str = "processed/index",
                 ollama_model: str = DEFAULT_OLLAMA_MODEL,
                 top_k: int = DEFAULT_TOP_K):
        """Initialize the retriever
        
        Args:
            model_name: Name of the embedding model to use
            index_folder: Folder containing the index
            ollama_model: Name of the Ollama model to use
            top_k: Number of chunks to retrieve
        """
        self.model_name = model_name
        self.index_folder = index_folder
        self.ollama_model = ollama_model
        self.top_k = top_k
        
        # Initialize model
        try:
            self.model = SentenceTransformer(model_name)
            print(f"Initialized embedding model: {model_name}")
        except Exception as e:
            print(f"Error initializing embedding model: {e}")
            raise
        
        # Load index and metadata
        self.index = None
        self.metadata = []
        self.corpus = []
        self._load_index()
    
    def _load_index(self) -> bool:
        """Load the index, corpus and metadata
        
        Returns:
            Success status
        """
        # Check if files exist
        index_file = os.path.join(self.index_folder, "faiss_index.idx")
        metadata_file = os.path.join(self.index_folder, "metadata.pkl")
        corpus_file = os.path.join(self.index_folder, "corpus.pkl")
        
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
    
    def check_ollama_running(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{OLLAMA_API_BASE}/tags")
            return response.status_code == 200
        except:
            return False
    
    def list_ollama_models(self) -> List[str]:
        """List available Ollama models"""
        try:
            response = requests.get(f"{OLLAMA_API_BASE}/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except:
            return []
    
    @track_function("Vector Search")
    def retrieve(self, query: ProcessedQuery) -> RetrievalResult:
        """Retrieve relevant chunks for a query
        
        Args:
            query: Processed query
            
        Returns:
            RetrievalResult object with relevant chunks
        """
        if self.index is None:
            raise ValueError("Index not loaded")
        
        # Start timing
        start_time = time.time()
        
        # Format query for embedding
        query_processor = QueryProcessor()
        formatted_query = query_processor.prepare_for_embedding(query)
        
        # Generate query embedding
        query_embedding = self.model.encode(
            [formatted_query], 
            normalize_embeddings=True
        )
        
        # Search index
        distances, indices = self.index.search(
            query_embedding.astype(np.float32),
            self.top_k
        )
        
        # Collect results
        chunks = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.corpus):  # Check for valid index
                chunk = RetrievedChunk(
                    text=self.corpus[idx],
                    metadata=self.metadata[idx],
                    distance=distances[0][i]
                )
                chunks.append(chunk)
        
        # Calculate retrieval time
        retrieval_time = time.time() - start_time
        
        return RetrievalResult(
            query=query.original_query,
            chunks=chunks,
            retrieval_time=retrieval_time
        )
    
    @track_function("LLM Generation")
    def generate_response(self, 
                        query: str, 
                        chunks: List[RetrievedChunk],
                        retrieval_time: float = 0.0,  # Add this parameter
                        temperature: float = DEFAULT_TEMPERATURE,
                        max_tokens: int = DEFAULT_MAX_TOKENS) -> ResponseResult:
        """Generate a response from the LLM
        
        Args:
            query: Original query
            chunks: Retrieved chunks
            retrieval_time: Time taken for retrieval
            temperature: Temperature for LLM generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            ResponseResult object with the answer
        """
        # Check if Ollama is running
        if not self.check_ollama_running():
            raise RuntimeError("Ollama is not running. Please start Ollama with 'ollama serve'")
        
        # Check if model is available
        available_models = self.list_ollama_models()
        if available_models and self.ollama_model not in available_models:
            print(f"Warning: Model '{self.ollama_model}' not found in Ollama. Available models: {', '.join(available_models)}")
            print(f"You may need to run: ollama pull {self.ollama_model}")
        
        # Start timing
        start_time = time.time()
        
        # Build context from chunks
        context = ""
        for i, chunk in enumerate(chunks):
            citation = chunk.get_citation()
            context += f"[Document {i+1}: {citation}]\n{chunk.text}\n\n"
        
        # Create prompt for Ollama
        prompt = f"""
You are a helpful assistant that provides accurate information based on the provided documents.

DOCUMENTS:
{context}

QUESTION:
{query}

Please provide a concise and accurate answer based solely on the information in the documents. 
If the documents don't contain the information needed to answer the question, say so clearly.
Please cite the specific documents you used in your answer using the document numbers [Document X].

ANSWER:
"""
        
        # Generate response from Ollama
        try:
            response = requests.post(
                f"{OLLAMA_API_BASE}/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                }
            )
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                answer = response.json().get("response", "No response from Ollama")
                
                # Record query metrics
                try:
                    query_metrics = QueryMetrics(
                        query=query,
                        query_length=len(query),
                        retrieval_time_seconds=0.0,  # Will be set in query method
                        llm_model=self.ollama_model,
                        llm_generation_time_seconds=generation_time,
                        response_length=len(answer),
                        num_chunks_retrieved=len(chunks),
                        peak_memory_mb=performance_tracker.get_peak_memory_mb(),
                        timestamp=datetime.now().isoformat()
                    )
                except:
                    pass  # Continue even if metrics recording fails
                
                # Fixed indentation - moved outside the except block
                return ResponseResult(
                    query=query,
                    answer=answer,
                    response_time=generation_time,
                    retrieval_time=retrieval_time,  # Now using the parameter
                    chunks_used=chunks,
                    model_name=self.ollama_model
                )
                
            else:
                error_msg = f"Error from Ollama API: {response.status_code} - {response.text}"
                return ResponseResult(
                    query=query,
                    answer=f"Error: {error_msg}",
                    response_time=generation_time,
                    retrieval_time=retrieval_time,  # Add the missing parameter
                    chunks_used=chunks,
                    model_name=self.ollama_model
                )
        
        except Exception as e:
            generation_time = time.time() - start_time
            error_msg = f"Error connecting to Ollama: {str(e)}"
            
            return ResponseResult(
                query=query,
                answer=f"Error: {error_msg}",
                response_time=generation_time,
                retrieval_time=retrieval_time,  # Add the missing parameter
                chunks_used=chunks,
                model_name=self.ollama_model
            )
    
    @track_function("Full Query")
    def query(self, 
              query: str, 
              temperature: float = DEFAULT_TEMPERATURE,
              max_tokens: int = DEFAULT_MAX_TOKENS) -> ResponseResult:
        """Process a query and generate a response
        
        Args:
            query: User query
            temperature: Temperature for LLM generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            ResponseResult object with the answer
        """
        # Start timing for the whole query process
        query_start_time = time.time()
        
        # Process query
        query_processor = QueryProcessor()
        processed_query = query_processor.process_query(query)
        
        # Retrieve relevant chunks
        retrieval_result = self.retrieve(processed_query)
        
        # Generate response
        response_result = self.generate_response(
            query, 
            retrieval_result.chunks,
            retrieval_result.retrieval_time,  # Pass the retrieval time
            temperature,
            max_tokens
        )
        
        # Record query metrics
        try:
            query_metrics = QueryMetrics(
                query=query,
                query_length=len(query),
                retrieval_time_seconds=retrieval_result.retrieval_time,
                llm_model=self.ollama_model,
                llm_generation_time_seconds=response_result.response_time,
                response_length=len(response_result.answer),
                num_chunks_retrieved=len(retrieval_result.chunks),
                peak_memory_mb=performance_tracker.get_peak_memory_mb(),
                timestamp=datetime.now().isoformat()
            )
            performance_tracker.record_query(query_metrics)
        except:
            print("Warning: Could not record query metrics")
        
        return response_result

# Example usage
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Query the PDF RAG system")
    parser.add_argument("query", nargs="?", type=str, help="Question to ask")
    parser.add_argument("--model", type=str, default=DEFAULT_OLLAMA_MODEL, help=f"Ollama model (default: {DEFAULT_OLLAMA_MODEL})")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"Temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help=f"Max tokens (default: {DEFAULT_MAX_TOKENS})")
    
    args = parser.parse_args()
    
    if not args.query:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Initialize retriever
        retriever = Retriever(
            ollama_model=args.model,
            top_k=args.top_k
        )
        
        # Process query
        result = retriever.query(
            args.query, 
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        # Print source information
        print("\nRetrieved chunks from:")
        for i, chunk in enumerate(result.chunks_used):
            citation = chunk.get_citation()
            print(f"  {i+1}. {citation} (distance: {chunk.distance:.4f})")
        
        # Print response
        print("\n=== RESPONSE ===")
        print(result.answer)
        print("===============\n")
        
        # Print performance
        print(f"Retrieval time: {result.retrieval_time:.2f} seconds")
        print(f"Generation time: {result.response_time:.2f} seconds")
        chars_per_second = len(result.answer) / result.response_time if result.response_time > 0 else 0
        print(f"Output speed: {chars_per_second:.2f} chars/second")
    
    except Exception as e:
        print(f"Error: {e}")