#!/usr/bin/env python3
# rag_app.py - Main application for the modular RAG system

"""
This script combines all modules in the RAG system:
1. Processes PDFs in the library folder
2. Creates chunks and embeddings
3. Allows querying with natural language

Usage:
  python3 rag_app.py process [--force] [--chunk-size SIZE] [--embedding-model MODEL]
  python3 rag_app.py query "Your question here" [--model MODEL] [--top-k K]
  python3 rag_app.py stats [--pdf | --query | --all]
"""

import os
import sys
import argparse
import json
import hashlib
from datetime import datetime

# Import modules
try:
    from pdf_processor import PDFProcessor
    from text_chunker import TextChunker
    from embedder import Embedder
    from query_processor import QueryProcessor
    from retriever import Retriever, ResponseResult
except ImportError as e:
    print(f"Error: Module not found: {e}")
    print("Make sure all modules are in the current directory.")
    sys.exit(1)

# Import performance tracking
try:
    from utils.performance_tracker import performance_tracker
    from view_performance import display_pdf_processing_metrics, display_query_metrics
except ImportError:
    print("Warning: Performance tracking not available")

# Constants
LIBRARY_FOLDER = "library"
PROCESSED_FOLDER = "processed"
PROCESSED_FILES_TRACKER = "processed_files.json"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_EMBEDDING_MODEL = "intfloat/e5-base-v2"
DEFAULT_OLLAMA_MODEL = "llama3:8b"
DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.6  # Add this
DEFAULT_MAX_TOKENS = 1000  # Add this

def ensure_directories():
    """Ensure all necessary directories exist"""
    directories = [
        LIBRARY_FOLDER,
        PROCESSED_FOLDER,
        os.path.join(PROCESSED_FOLDER, "index"),
        os.path.join(PROCESSED_FOLDER, "chunks"),
        os.path.join(PROCESSED_FOLDER, "embeddings")
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_pdf_hash(file_path):
    """Generate a hash for a PDF file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_processed_files():
    """Load the list of processed files"""
    try:
        if os.path.exists(PROCESSED_FILES_TRACKER):
            with open(PROCESSED_FILES_TRACKER, "r") as f:
                content = f.read().strip()
                return json.loads(content) if content else {"processed_files": []}
    except json.JSONDecodeError:
        print(f"Warning: Could not decode {PROCESSED_FILES_TRACKER}. Starting fresh.")
    return {"processed_files": []}

def save_processed_files(processed_files):
    """Save the list of processed files"""
    with open(PROCESSED_FILES_TRACKER, "w") as f:
        json.dump(processed_files, f, indent=2)

def save_chunks_and_embeddings(chunks, pdf_path, document):
    """Save chunks and embeddings to disk."""
    # Create directories if they don't exist
    chunks_dir = os.path.join(PROCESSED_FOLDER, "chunks")
    embeddings_dir = os.path.join(PROCESSED_FOLDER, "embeddings")
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Use the correct hash attribute based on the document object
    doc_hash = document.hash if hasattr(document, 'hash') else document.doc_hash
    
    # Save chunks
    chunks_file = os.path.join(chunks_dir, f"{doc_hash}_chunks.pkl")
    with open(chunks_file, "wb") as f:
        import pickle
        pickle.dump(chunks, f)
    
    # Extract embeddings from chunks if available
    try:
        embeddings = [chunk.embedding for chunk in chunks if hasattr(chunk, 'embedding')]
        if embeddings:
            embeddings_file = os.path.join(embeddings_dir, f"{doc_hash}_embeddings.npy")
            import numpy as np
            np.save(embeddings_file, embeddings)
            print(f"Saved chunks and embeddings for {pdf_path}")
        else:
            print(f"Saved chunks for {pdf_path} (no embeddings found in chunks)")
    except Exception as e:
        print(f"Error saving embeddings: {e}")

def process_command(args):
    """Comprehensive PDF processing command"""
    # Ensure all necessary directories exist
    ensure_directories()
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(LIBRARY_FOLDER) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in {LIBRARY_FOLDER} folder.")
        return
    
    print(f"Found {len(pdf_files)} PDF files in the library.")
    
    # Load processed files tracker
    processed_files = load_processed_files()
    
    # Initialize components
    pdf_processor = PDFProcessor()
    text_chunker = TextChunker(chunk_size=args.chunk_size)
    embedder = Embedder(
        model_name=args.embedding_model, 
        batch_size=args.batch_size
    )
    
    # Process each PDF
    new_pdfs_processed = 0
    total_chunks = 0
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(LIBRARY_FOLDER, pdf_file)
        
        # Check if already processed
        pdf_hash = get_pdf_hash(pdf_path)
        already_processed = any(
            item.get("pdf_hash") == pdf_hash 
            for item in processed_files.get("processed_files", [])
        )
        
        if already_processed and not args.force:
            print(f"Skipping already processed file: {pdf_file}")
            continue
            
        print(f"\nProcessing: {pdf_file}")
        
        try:
            # Process PDF
            document = pdf_processor.process(pdf_path)
            
            # Let's add some debugging to understand the document structure
            print(f"Document object attributes: {dir(document)}")
            
            # Create chunks
            chunks = text_chunker.chunk_document(document)
            
            if not chunks:
                print(f"Warning: No chunks created for {pdf_file}")
                continue
            
            print(f"Created {len(chunks)} chunks from {document.filename}")
            
            # Add chunks to embedder and save index
            embedder.add_chunks(chunks)
            embedder.save_index()
            
            # Save chunks and embeddings to their respective folders
            save_chunks_and_embeddings(chunks, pdf_path, document)
            
            # Update processed files list
            if not already_processed:
                processed_files.setdefault("processed_files", []).append({
                    "pdf_path": pdf_path,
                    "pdf_hash": document.hash,
                    "processed_time": document.processed_time,
                    "num_chunks": len(chunks)
                })
            elif args.force:
                # Update existing entry
                for item in processed_files.get("processed_files", []):
                    if item.get("pdf_hash") == document.hash:
                        item["processed_time"] = document.processed_time
                        item["num_chunks"] = len(chunks)
                        break
            
            new_pdfs_processed += 1
            total_chunks += len(chunks)
            
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save processed files list
    save_processed_files(processed_files)
    
    # Print summary
    if new_pdfs_processed > 0:
        print(f"\nProcessed {new_pdfs_processed} new PDF files")
        print(f"Total chunks created: {total_chunks}")
        print(f"Average chunks per PDF: {total_chunks / new_pdfs_processed:.2f}")
    else:
        print("\nNo new PDF files to process")
    
    # Show statistics if requested
    if args.show_stats:
        try:
            display_pdf_processing_metrics()
        except Exception as e:
            print(f"Error displaying statistics: {e}")

def query_command(args):
    """Process a query against the RAG system"""
    # Ensure all necessary directories exist
    ensure_directories()
    
    question = args.question
    print(f"Processing query: {question}")
    
    # Initialize components
    try:
        retriever = Retriever(
            ollama_model=args.model,
            top_k=args.top_k
        )
        
        # Process query
        result = retriever.query(question)
        
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
        import traceback
        traceback.print_exc()

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process PDFs in the library")
    process_parser.add_argument("--force", action="store_true", help="Force reprocessing of all PDFs")
    process_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size for text splitting")
    process_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model to use")
    process_parser.add_argument("--batch-size", type=int, default=8, help="Batch size for embeddings")
    process_parser.add_argument("--show-stats", action="store_true", help="Show processing statistics")
    
    # Query command
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="LLM model to use")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    query_parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Temperature for generation")
    query_parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum tokens to generate")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show system statistics")
    stats_parser.add_argument("--pdf", action="store_true", help="Show PDF processing statistics")
    stats_parser.add_argument("--query", action="store_true", help="Show query statistics")
    stats_parser.add_argument("--all", action="store_true", help="Show all statistics")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "process":
        process_command(args)
    elif args.command == "query":
        query_command(args)
    elif args.command == "stats":
        # Implement stats command here
        pass
    else:
        parser.print_help()

if __name__ == "__main__":
    main()