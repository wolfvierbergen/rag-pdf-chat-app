#!/usr/bin/env python3
# debug_processing.py - Debug tool for document processing

import os
import sys
import json
import time
from pprint import pprint

def debug_processing_pipeline(pdf_path):
    """Debug the entire document processing pipeline"""
    print(f"\n===== Debugging PDF Processing Pipeline =====")
    print(f"PDF: {pdf_path}")
    
    try:
        from pdf_processor import PDFProcessor
        from text_chunker import TextChunker
        
        # Step 1: Process PDF
        print("\n=== Step 1: PDF Processing ===")
        processor = PDFProcessor()
        
        start_time = time.time()
        document = processor.process(pdf_path)
        pdf_process_time = time.time() - start_time
        
        print(f"PDF processed in {pdf_process_time:.2f} seconds")
        print(f"Document class: {document.__class__.__name__}")
        
        # Check document attributes
        print("\nDocument attributes:")
        for attr in ["filename", "path", "hash", "processed_time"]:
            if hasattr(document, attr):
                print(f"  {attr}: {getattr(document, attr)}")
            else:
                print(f"  {attr}: [Not found]")
        
        # Find content attribute
        content_attr = None
        for attr in ["content", "extracted_text", "text", "fulltext"]:
            if hasattr(document, attr):
                content = getattr(document, attr)
                if isinstance(content, str) and len(content) > 0:
                    content_attr = attr
                    content_preview = content[:100] + "..." if len(content) > 100 else content
                    print(f"\nFound text content in '{attr}' attribute: {content_preview}")
                    print(f"Content length: {len(content)} characters")
                    break
        
        if not content_attr:
            print("\nERROR: No text content found in document!")
            print("Available attributes:")
            for attr in dir(document):
                if not attr.startswith('__'):
                    try:
                        value = getattr(document, attr)
                        print(f"  {attr}: {type(value)}")
                    except:
                        pass
            return
        
        # Step 2: Text Chunking
        print("\n=== Step 2: Text Chunking ===")
        chunker = TextChunker(chunk_size=500)
        
        # Patch for accessing content
        document.text = getattr(document, content_attr)
        
        start_time = time.time()
        chunks = chunker.chunk_document(document)
        chunk_time = time.time() - start_time
        
        print(f"Created {len(chunks)} chunks in {chunk_time:.2f} seconds")
        
        if chunks:
            print("\nFirst chunk preview:")
            chunk = chunks[0]
            print(f"  Text: {chunk.text[:100]}...")
            print(f"  Token count: {chunk.token_count}")
            print(f"  Metadata: doc_title={chunk.doc_title}, doc_hash={chunk.doc_hash}")
            
            print("\nLast chunk preview:")
            chunk = chunks[-1]
            print(f"  Text: {chunk.text[:100]}...")
            
            print(f"\nAverage chunk size: {sum(len(c.text) for c in chunks)/len(chunks):.1f} characters")
            print(f"Average token count: {sum(c.token_count for c in chunks)/len(chunks):.1f} tokens")
        
        print("\nProcessing complete!")
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_processing.py /path/to/document.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    success = debug_processing_pipeline(pdf_path)
    if not success:
        sys.exit(1)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="RAG System for PDF Documents")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process PDFs in the library folder")
    process_parser.add_argument("--force", action="store_true", 
                                help="Force reprocessing of all PDFs")
    process_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, 
                                help=f"Token size for chunking (default: {DEFAULT_CHUNK_SIZE})")
    process_parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL, 
                                help=f"Embedding model to use (default: {DEFAULT_EMBEDDING_MODEL})")
    process_parser.add_argument("--batch-size", type=int, default=32, 
                                help="Batch size for embedding generation (default: 32)")
    process_parser.add_argument("--show-stats", action="store_true", 
                                help="Show performance statistics")
    
    # Query command - add other command parsers here if needed
    # query_parser = subparsers.add_parser("query", help="Query the RAG system")
    # ...
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "process":
        process_command(args)
    elif args.command == "query":
        # query_command(args)
        pass
    # Add other command handlers as needed
    else:
        parser.print_help()

if __name__ == "__main__":
    main()