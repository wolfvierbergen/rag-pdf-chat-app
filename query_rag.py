#!/usr/bin/env python3
# query_rag.py - Query the PDF RAG system using Ollama with performance tracking

"""
This script queries the PDF RAG system using Ollama:
1. Converts the question to an embedding
2. Retrieves the most relevant chunks from the FAISS index
3. Passes the chunks to Ollama for a response
4. Tracks performance metrics for all operations

Usage:
  python query_rag.py "your question here" [--model MODEL_NAME] [--top-k 3]

Arguments:
  question         The question to ask the RAG system
  --model          Ollama model to use (default: llama3)
  --top-k          Number of chunks to retrieve (default: 3)
  --temperature    Temperature for generation (default: 0.7)
  --max-tokens     Maximum number of tokens to generate (default: 500)
  --show-stats     Show performance statistics for all queries
"""

import os
import sys
import time
import json
import pickle
import argparse
import requests
import numpy as np
import faiss
from pathlib import Path

# Try to import required libraries
try:
    from sentence_transformers import SentenceTransformer
    import psutil
except ImportError:
    print("Error: Required libraries not installed.")
    print("Please install the required libraries with:")
    print("pip install sentence-transformers faiss-cpu requests numpy psutil")
    sys.exit(1)

# Import performance tracking utilities
try:
    from utils.performance_tracker import performance_tracker, QueryMetrics, track_function
except ImportError:
    print("Error: Performance tracking utilities not found.")
    print("Please ensure the performance_tracker.py file exists in the utils directory.")
    sys.exit(1)

# Constants
INDEX_FOLDER = "processed/index"
METADATA_FILE = f"{INDEX_FOLDER}/metadata.pkl"
INDEX_FILE = f"{INDEX_FOLDER}/faiss_index.idx"
CORPUS_FILE = f"{INDEX_FOLDER}/corpus.pkl"
EMBEDDING_MODEL = "intfloat/e5-base-v2"
DEFAULT_OLLAMA_MODEL = "llama3:8b"  # Default Ollama model
OLLAMA_API_BASE = "http://localhost:11434/api"
DEFAULT_TOP_K = 3
DEFAULT_TEMPERATURE = 0.6
DEFAULT_MAX_TOKENS = 500

@track_function("Loading RAG Components")
def load_rag_components():
    """Load the corpus, embeddings, metadata, and FAISS index."""
    try:
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)
        
        with open(CORPUS_FILE, "rb") as f:
            corpus = pickle.load(f)
        
        index = faiss.read_index(INDEX_FILE)
        
        return corpus, metadata, index
    except FileNotFoundError as e:
        print(f"Error loading RAG components: {e}")
        print("Please process some PDFs first with process_pdfs.py.")
        sys.exit(1)

@track_function("Query Embedding")
def generate_query_embedding(query, embedding_model):
    """Convert a query to an embedding."""
    query_embedding = embedding_model.encode(
        [f"query: {query}"], normalize_embeddings=True
    )
    return query_embedding

@track_function("Chunk Retrieval")
def retrieve_relevant_chunks(query, embedding_model, index, corpus, metadata, top_k=DEFAULT_TOP_K):
    """Retrieve the most relevant chunks for a query."""
    retrieval_start = time.time()
    
    # Generate query embedding
    query_embedding = generate_query_embedding(query, embedding_model)
    
    # Search for similar chunks
    distances, indices = index.search(
        query_embedding.astype(np.float32),
        top_k
    )
    
    # Get the relevant chunks and their metadata
    relevant_chunks = []
    for i, idx in enumerate(indices[0]):
        chunk = corpus[idx]
        chunk_metadata = metadata[idx]
        pdf_name = os.path.basename(chunk_metadata["pdf_path"])
        
        relevant_chunks.append({
            "chunk": chunk,
            "distance": distances[0][i],
            "metadata": chunk_metadata,
            "pdf_name": pdf_name
        })
    
    retrieval_time = time.time() - retrieval_start
    
    return relevant_chunks, retrieval_time

def check_ollama_running():
    """Check if Ollama is running."""
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/tags")
        if response.status_code == 200:
            return True
    except requests.exceptions.ConnectionError:
        pass
    return False

def list_ollama_models():
    """List available Ollama models."""
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return []
    except:
        return []

@track_function("LLM Generation")
def generate_ollama_response(query, relevant_chunks, model_name=DEFAULT_OLLAMA_MODEL, 
                            temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS):
    """Generate a response from Ollama based on the query and relevant chunks."""
    generation_start = time.time()
    
    if not check_ollama_running():
        print("Error: Ollama is not running. Please start Ollama with 'ollama serve'")
        sys.exit(1)
    
    available_models = list_ollama_models()
    if not available_models:
        print("Warning: Could not retrieve list of available models from Ollama")
    elif model_name not in available_models:
        print(f"Warning: Model '{model_name}' not found in Ollama. Available models: {', '.join(available_models)}")
        print(f"You may need to run: ollama pull {model_name}")
    
    print(f"Using Ollama model: {model_name}")
    
    # Build context from chunks
    context = ""
    for i, item in enumerate(relevant_chunks):
        context += f"[Document {i+1}: {item['pdf_name']}]\n{item['chunk']}\n\n"
    
    # Create prompt for Ollama
    prompt = f"""
You are a helpful assistant that provides accurate information based on the provided documents.

DOCUMENTS:
{context}

QUESTION:
{query}

Please provide a concise and accurate answer based solely on the information in the documents. If the documents don't contain the information needed to answer the question, say so clearly.

ANSWER:
"""
    
    # Generate response from Ollama
    print("Generating response from Ollama...")
    try:
        response = requests.post(
            f"{OLLAMA_API_BASE}/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
        )
        
        generation_time = time.time() - generation_start
        
        if response.status_code == 200:
            answer = response.json().get("response", "No response from Ollama")
            return answer, generation_time, len(answer)
        else:
            error_msg = f"Error from Ollama API: {response.status_code} - {response.text}"
            print(error_msg)
            return f"Error: {error_msg}", generation_time, 0
    
    except Exception as e:
        generation_time = time.time() - generation_start
        error_msg = f"Error connecting to Ollama: {str(e)}"
        print(error_msg)
        return f"Error: {error_msg}", generation_time, 0

def main():
    parser = argparse.ArgumentParser(description="Query the PDF RAG system using Ollama")
    parser.add_argument("question", type=str, nargs="?", help="The question to ask")
    parser.add_argument("--model", type=str, default=DEFAULT_OLLAMA_MODEL, 
                        help=f"Ollama model to use (default: {DEFAULT_OLLAMA_MODEL})")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, 
                        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, 
                        help=f"Temperature for generation (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, 
                        help=f"Maximum tokens to generate (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--show-stats", action="store_true", 
                        help="Show performance statistics for all queries")
    
    args = parser.parse_args()
    
    # Show stats if requested
    if args.show_stats:
        stats = performance_tracker.get_query_stats()
        print("\n===== Query Performance Statistics =====")
        for key, value in stats.items():
            if key != "models":
                if isinstance(value, float):
                    print(f"{key}: {value:.2f}")
                else:
                    print(f"{key}: {value}")
        
        print("\nPerformance by Model:")
        for model, model_stats in stats.get("models", {}).items():
            print(f"  {model}:")
            for stat_key, stat_value in model_stats.items():
                if isinstance(stat_value, float):
                    print(f"    {stat_key}: {stat_value:.2f}")
                else:
                    print(f"    {stat_key}: {stat_value}")
        sys.exit(0)
    
    # Ensure question is provided
    if not args.question:
        parser.print_help()
        sys.exit(1)
    
    # Check if index exists
    if not all(os.path.exists(f) for f in [METADATA_FILE, INDEX_FILE, CORPUS_FILE]):
        print("Error: RAG index not found. Please process some PDFs first with process_pdfs.py.")
        sys.exit(1)
    
    # Load embedding model
    try:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        sys.exit(1)
    
    # Load RAG components
    corpus, metadata, index = load_rag_components()
    
    print(f"Loaded index with {index.ntotal} vectors")
    
    # Retrieve relevant chunks with performance tracking
    print(f"Searching for relevant chunks for query: '{args.question}'")
    relevant_chunks, retrieval_time = retrieve_relevant_chunks(
        args.question, embedding_model, index, corpus, metadata, args.top_k
    )
    
    # Print source information
    print("\nRetrieved chunks from:")
    for i, item in enumerate(relevant_chunks):
        print(f"  {i+1}. {item['pdf_name']} (distance: {item['distance']:.4f})")
    
    # Generate response using Ollama with performance tracking
    response, generation_time, response_length = generate_ollama_response(
        args.question, relevant_chunks, args.model, 
        args.temperature, args.max_tokens
    )
    
    # Record query metrics
    query_metrics = QueryMetrics(
        query=args.question,
        query_length=len(args.question),
        retrieval_time_seconds=retrieval_time,
        llm_model=args.model,
        llm_generation_time_seconds=generation_time,
        response_length=response_length,
        num_chunks_retrieved=len(relevant_chunks),
        peak_memory_mb=performance_tracker.get_peak_memory_mb(),
        timestamp=datetime.now().isoformat()
    )
    performance_tracker.record_query(query_metrics)
    
    # Print response
    print("\n=== RESPONSE ===")
    print(response)
    print("===============\n")
    
    # Print performance summary
    chars_per_second = response_length / generation_time if generation_time > 0 else 0
    print(f"Query performance:")
    print(f"  Retrieval time: {retrieval_time:.2f} seconds")
    print(f"  Generation time: {generation_time:.2f} seconds")
    print(f"  Output speed: {chars_per_second:.2f} chars/second")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total query time: {elapsed_time:.2f} seconds")