#!/usr/bin/env python3
# process_pdfs.py - Process PDFs in the library folder

"""
This script processes PDFs in the 'library' folder:
1. Extracts text from new PDFs (those not previously processed)
2. Chunks the text into smaller pieces
3. Creates embeddings for the chunks
4. Updates the FAISS index

Usage:
  python process_pdfs.py [--force] [--chunk-size 500]

Arguments:
  --force           Force reprocessing of all PDFs
  --chunk-size      Token size for chunking (default: 500)
"""

import os
import sys
import json
import time
import hashlib
import argparse
import numpy as np
import faiss
import pickle
from pathlib import Path
from datetime import datetime

# Try to import required libraries
try:
    import tiktoken
    from PyPDF2 import PdfReader
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: Required libraries not installed.")
    print("Please install the required libraries with:")
    print("pip install PyPDF2 tiktoken sentence-transformers faiss-cpu")
    sys.exit(1)

# Constants
LIBRARY_FOLDER = "library"
PROCESSED_FOLDER = "processed"
CHUNKS_FOLDER = f"{PROCESSED_FOLDER}/chunks"
EMBEDDINGS_FOLDER = f"{PROCESSED_FOLDER}/embeddings"
INDEX_FOLDER = f"{PROCESSED_FOLDER}/index"
PROCESSED_FILES_TRACKER = "processed_files.json"
TOKENIZER_NAME = "o200k_base"
EMBEDDING_MODEL = "intfloat/e5-base-v2"
DEFAULT_CHUNK_SIZE = 500
METADATA_FILE = f"{INDEX_FOLDER}/metadata.pkl"
INDEX_FILE = f"{INDEX_FOLDER}/faiss_index.idx"
CORPUS_FILE = f"{INDEX_FOLDER}/corpus.pkl"

def get_pdf_hash(file_path):
    """Create a hash of the PDF file to uniquely identify it."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_processed_files():
    """Load the list of processed files."""
    if os.path.exists(PROCESSED_FILES_TRACKER):
        with open(PROCESSED_FILES_TRACKER, "r") as f:
            return json.load(f)
    else:
        return {"processed_files": []}

def save_processed_files(processed_files):
    """Save the list of processed files."""
    with open(PROCESSED_FILES_TRACKER, "w") as f:
        json.dump(processed_files, f, indent=2)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    print(f"Extracting text from: {pdf_path}")
    
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def chunk_text(text, tokenizer, max_tokens=DEFAULT_CHUNK_SIZE):
    """Split text into chunks of specified token size."""
    tokens = tokenizer.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunks.append(tokenizer.decode(tokens[i:i + max_tokens]))
        i += max_tokens
    return chunks

def process_pdf(pdf_path, tokenizer, chunk_size=DEFAULT_CHUNK_SIZE):
    """Process a PDF file: extract text and create chunks."""
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print(f"Warning: No text extracted from {pdf_path}")
        return []
    
    # Chunk the text
    print(f"Chunking text into {chunk_size}-token chunks...")
    chunks = chunk_text(text, tokenizer, chunk_size)
    print(f"Created {len(chunks)} chunks")
    
    return chunks

def generate_embeddings(chunks, embedding_model):
    """Generate embeddings for text chunks."""
    print(f"Generating embeddings using {EMBEDDING_MODEL}...")
    passages = [f"passage: {chunk}" for chunk in chunks]
    
    embeddings = embedding_model.encode(
        passages, normalize_embeddings=True, 
        show_progress_bar=True
    )
    
    return embeddings

def update_faiss_index(new_embeddings, new_chunks, pdf_info, metadata=None, existing_index=None, existing_corpus=None):
    """Update or create FAISS index with new embeddings."""
    if metadata is None:
        metadata = []
    
    if existing_corpus is None:
        corpus = []
    else:
        corpus = existing_corpus.copy()
    
    # Create new metadata entries for all new chunks
    new_metadata = []
    for i in range(len(new_chunks)):
        chunk_metadata = {
            "pdf_path": pdf_info["pdf_path"],
            "pdf_hash": pdf_info["pdf_hash"],
            "chunk_index": i,
            "processed_time": pdf_info["processed_time"],
            "chunk_id": f"{pdf_info['pdf_hash']}_{i}"
        }
        new_metadata.append(chunk_metadata)
    
    # Update metadata and corpus
    start_idx = len(metadata)
    metadata.extend(new_metadata)
    corpus.extend(new_chunks)
    
    # Create or update the FAISS index
    if existing_index is None:
        # Create new index
        dimension = new_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(new_embeddings.astype(np.float32))
        print(f"Created new FAISS index with {len(new_chunks)} vectors")
    else:
        # Update existing index
        index = existing_index
        index.add(new_embeddings.astype(np.float32))
        print(f"Updated FAISS index, added {len(new_chunks)} new vectors (now contains {index.ntotal} total)")
    
    # Save updated metadata, corpus, and index
    os.makedirs(INDEX_FOLDER, exist_ok=True)
    
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)
    
    with open(CORPUS_FILE, "wb") as f:
        pickle.dump(corpus, f)
    
    faiss.write_index(index, INDEX_FILE)
    
    # Return updated data
    return index, corpus, metadata

def load_existing_index():
    """Load existing FAISS index and metadata if they exist."""
    if not (os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE) and os.path.exists(CORPUS_FILE)):
        return None, None, None
    
    try:
        index = faiss.read_index(INDEX_FILE)
        
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)
        
        with open(CORPUS_FILE, "rb") as f:
            corpus = pickle.load(f)
        
        print(f"Loaded existing index with {index.ntotal} vectors and {len(metadata)} metadata entries")
        return index, corpus, metadata
    except Exception as e:
        print(f"Error loading existing index: {e}")
        return None, None, None

def save_chunks_and_embeddings(chunks, embeddings, pdf_info):
    """Save chunks and embeddings to disk."""
    pdf_hash = pdf_info["pdf_hash"]
    
    # Create directories if they don't exist
    os.makedirs(CHUNKS_FOLDER, exist_ok=True)
    os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)
    
    # Save chunks
    chunks_file = f"{CHUNKS_FOLDER}/{pdf_hash}_chunks.pkl"
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)
    
    # Save embeddings
    embeddings_file = f"{EMBEDDINGS_FOLDER}/{pdf_hash}_embeddings.npy"
    np.save(embeddings_file, embeddings)
    
    print(f"Saved chunks and embeddings for {pdf_info['pdf_path']}")

def main():
    parser = argparse.ArgumentParser(description="Process PDFs in the library folder")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all PDFs")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help=f"Token size for chunking (default: {DEFAULT_CHUNK_SIZE})")
    
    args = parser.parse_args()
    
    # Check if required folders exist
    for folder in [LIBRARY_FOLDER, PROCESSED_FOLDER, CHUNKS_FOLDER, EMBEDDINGS_FOLDER, INDEX_FOLDER]:
        if not os.path.exists(folder):
            print(f"Error: Required folder {folder} does not exist. Please run project_structure.py first.")
            sys.exit(1)
    
    # Load the list of processed files
    processed_files = load_processed_files()
    
    # Get list of PDF files in the library folder
    pdf_files = [f for f in os.listdir(LIBRARY_FOLDER) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in {LIBRARY_FOLDER} folder.")
        sys.exit(0)
    
    print(f"Found {len(pdf_files)} PDF files in the library.")
    
    # Initialize tokenizer
    try:
        tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    except Exception as e:
        print(f"Error initializing tokenizer: {e}")
        sys.exit(1)
    
    # Initialize embedding model
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        sys.exit(1)
    
    # Load existing index and metadata
    existing_index, existing_corpus, existing_metadata = load_existing_index()
    
    # Process each PDF file that hasn't been processed yet
    new_pdfs_processed = 0
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(LIBRARY_FOLDER, pdf_file)
        pdf_hash = get_pdf_hash(pdf_path)
        
        # Check if this PDF has already been processed
        already_processed = any(item["pdf_hash"] == pdf_hash for item in processed_files["processed_files"])
        
        if already_processed and not args.force:
            print(f"Skipping already processed file: {pdf_file}")
            continue
        
        print(f"\nProcessing: {pdf_file}")
        
        # Process the PDF
        chunks = process_pdf(pdf_path, tokenizer, args.chunk_size)
        
        if not chunks:
            print(f"Skipping {pdf_file} - no text chunks extracted")
            continue
        
        # Generate embeddings
        embeddings = generate_embeddings(chunks, embedding_model)
        
        # Create PDF info
        pdf_info = {
            "pdf_path": pdf_path,
            "pdf_hash": pdf_hash,
            "processed_time": datetime.now().isoformat(),
            "num_chunks": len(chunks)
        }
        
        # Save chunks and embeddings
        save_chunks_and_embeddings(chunks, embeddings, pdf_info)
        
        # Update FAISS index
        existing_index, existing_corpus, existing_metadata = update_faiss_index(
            embeddings, chunks, pdf_info, 
            existing_metadata, existing_index, existing_corpus
        )
        
        # Add to processed files list if not already there
        if not already_processed:
            processed_files["processed_files"].append({
                "pdf_path": pdf_path,
                "pdf_hash": pdf_hash,
                "processed_time": pdf_info["processed_time"],
                "num_chunks": len(chunks)
            })
        elif args.force:
            # Update the entry if reprocessing
            for item in processed_files["processed_files"]:
                if item["pdf_hash"] == pdf_hash:
                    item["processed_time"] = pdf_info["processed_time"]
                    item["num_chunks"] = len(chunks)
                    break
        
        new_pdfs_processed += 1
    
    # Save the updated processed files list
    save_processed_files(processed_files)
    
    if new_pdfs_processed > 0:
        print(f"\nProcessed {new_pdfs_processed} new PDF files")
    else:
        print("\nNo new PDF files to process")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")#!/usr/bin/env python

