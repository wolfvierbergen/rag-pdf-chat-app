#!/usr/bin/env python3
# check_processed_files.py - Utility to check processed files status

import os
import json
import pickle
import numpy as np
from pathlib import Path

# Constants
PROCESSED_FOLDER = "processed"
CHUNKS_FOLDER = f"{PROCESSED_FOLDER}/chunks"
EMBEDDINGS_FOLDER = f"{PROCESSED_FOLDER}/embeddings"
INDEX_FOLDER = f"{PROCESSED_FOLDER}/index"
PROCESSED_FILES_TRACKER = "processed_files.json"

def check_processed_files():
    """Check processed files and their corresponding chunks/embeddings"""
    print("\n===== Checking Processed Files Status =====\n")
    
    # Check if processed files tracker exists
    if not os.path.exists(PROCESSED_FILES_TRACKER):
        print(f"Error: {PROCESSED_FILES_TRACKER} not found!")
        return
    
    # Load processed files
    with open(PROCESSED_FILES_TRACKER, "r") as f:
        try:
            processed_files = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: {PROCESSED_FILES_TRACKER} is corrupted!")
            return
    
    if not processed_files.get("processed_files"):
        print("No processed files found in tracker.")
        return
    
    print(f"Found {len(processed_files['processed_files'])} processed files in tracker.")
    
    # Check if folders exist
    for folder in [CHUNKS_FOLDER, EMBEDDINGS_FOLDER, INDEX_FOLDER]:
        if not os.path.exists(folder):
            print(f"Warning: {folder} directory doesn't exist!")
            os.makedirs(folder, exist_ok=True)
            print(f"Created {folder} directory.")
    
    # Check index files
    index_files = [
        os.path.join(INDEX_FOLDER, "faiss_index.idx"), 
        os.path.join(INDEX_FOLDER, "metadata.pkl"),
        os.path.join(INDEX_FOLDER, "corpus.pkl")
    ]
    
    for file in index_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✓ {file} exists ({size/1024:.2f} KB)")
        else:
            print(f"✗ {file} missing!")
    
    # Check chunks and embeddings
    chunks_count = len([f for f in os.listdir(CHUNKS_FOLDER) if f.endswith("_chunks.pkl")])
    embeddings_count = len([f for f in os.listdir(EMBEDDINGS_FOLDER) if f.endswith("_embeddings.npy")])
    
    print(f"\nFound {chunks_count} chunk files and {embeddings_count} embedding files.")
    
    # Check each processed file
    print("\nIndividual file check:")
    print("-" * 80)
    print(f"{'PDF Hash':<12} | {'Chunks':<8} | {'Embeddings':<10} | {'In Index':<8}")
    print("-" * 80)
    
    for item in processed_files["processed_files"]:
        pdf_hash = item.get("pdf_hash", "unknown")
        
        # Check chunks
        chunks_file = os.path.join(CHUNKS_FOLDER, f"{pdf_hash}_chunks.pkl")
        chunks_status = "✓" if os.path.exists(chunks_file) else "✗"
        
        # Check embeddings
        embeddings_file = os.path.join(EMBEDDINGS_FOLDER, f"{pdf_hash}_embeddings.npy")
        embeddings_status = "✓" if os.path.exists(embeddings_file) else "✗"
        
        # Check if in index
        in_index = "Unknown"
        if os.path.exists(os.path.join(INDEX_FOLDER, "metadata.pkl")):
            try:
                with open(os.path.join(INDEX_FOLDER, "metadata.pkl"), "rb") as f:
                    metadata = pickle.load(f)
                    # Check if any metadata entry has this doc_hash
                    in_index = "✓" if any(m.get("doc_hash") == pdf_hash for m in metadata) else "✗"
            except:
                in_index = "Error"
        
        print(f"{pdf_hash[:10]:<12} | {chunks_status:<8} | {embeddings_status:<10} | {in_index:<8}")
    
    print("\nChecking index integrity...")
    # Load index and check dimensions
    try:
        if os.path.exists(os.path.join(INDEX_FOLDER, "faiss_index.idx")):
            import faiss
            index = faiss.read_index(os.path.join(INDEX_FOLDER, "faiss_index.idx"))
            print(f"✓ Index contains {index.ntotal} vectors with dimension {index.d}")
            
            # Load metadata and corpus to check consistency
            if os.path.exists(os.path.join(INDEX_FOLDER, "metadata.pkl")) and os.path.exists(os.path.join(INDEX_FOLDER, "corpus.pkl")):
                with open(os.path.join(INDEX_FOLDER, "metadata.pkl"), "rb") as f:
                    metadata = pickle.load(f)
                with open(os.path.join(INDEX_FOLDER, "corpus.pkl"), "rb") as f:
                    corpus = pickle.load(f)
                
                if len(metadata) == index.ntotal and len(corpus) == index.ntotal:
                    print(f"✓ Index, metadata, and corpus are in sync ({len(metadata)} entries)")
                else:
                    print(f"✗ Index, metadata, and corpus are out of sync:")
                    print(f"  - Index: {index.ntotal} entries")
                    print(f"  - Metadata: {len(metadata)} entries")
                    print(f"  - Corpus: {len(corpus)} entries")
        else:
            print("✗ Index file doesn't exist")
    except Exception as e:
        print(f"Error checking index: {e}")
    
    print("\nDone!")

if __name__ == "__main__":
    check_processed_files()
