#!/usr/bin/env python3
# project_structure.py - Creates the folder structure for the PDF RAG system

"""
This script sets up the required folder structure for the PDF RAG system:

- library/              # Where you drop your PDFs
- processed/            # Where processed data is stored
  - chunks/             # Text chunks extracted from PDFs
  - embeddings/         # Embeddings for each document
  - index/              # FAISS index and metadata

Usage:
  python project_structure.py
"""

import os
import json
import sys

# Define folder structure
FOLDERS = [
    "library",
    "processed",
    "processed/chunks",
    "processed/embeddings",
    "processed/index"
]

# Create a processed_files.json tracker file
PROCESSED_FILES_TRACKER = "processed_files.json"

def setup_project_structure():
    """Create the folder structure for the PDF RAG system."""
    print("Setting up PDF RAG system folder structure...")
    
    # Create folders
    for folder in FOLDERS:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")
        else:
            print(f"Folder already exists: {folder}")
    
    # Create processed files tracker if it doesn't exist
    if not os.path.exists(PROCESSED_FILES_TRACKER):
        with open(PROCESSED_FILES_TRACKER, "w") as f:
            json.dump({"processed_files": []}, f)
        print(f"Created file: {PROCESSED_FILES_TRACKER}")
    else:
        print(f"File already exists: {PROCESSED_FILES_TRACKER}")
    
    print("\nFolder structure setup complete!")
    print("\nUsage:")
    print("1. Place your PDF files in the 'library' folder")
    print("2. Run 'python process_pdfs.py' to process new PDFs")
    print("3. Run 'python query_rag.py \"your question\"' to query the system")

if __name__ == "__main__":
    setup_project_structure()#!/usr/bin/env python

