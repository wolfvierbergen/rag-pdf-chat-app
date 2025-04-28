#!/usr/bin/env python3
# check_document.py - Utility to inspect document objects

import sys
from pprint import pprint

def check_document(pdf_path):
    """Examine the document object from a PDF file"""
    try:
        from pdf_processor import PDFProcessor
        
        processor = PDFProcessor()
        document = processor.process(pdf_path)
        
        print("\n===== Document Object Inspection =====\n")
        
        # Print all attributes
        print("Document attributes:")
        for attr in dir(document):
            if not attr.startswith('__'):
                try:
                    value = getattr(document, attr)
                    # For large text content, just show the beginning
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "... [truncated]"
                    print(f"  {attr}: {value}")
                except:
                    print(f"  {attr}: [Error accessing attribute]")
        
        # Check if content is accessible
        content_attrs = ["text", "content", "extracted_text", "fulltext"]
        print("\nTrying to access text content:")
        for attr in content_attrs:
            try:
                if hasattr(document, attr):
                    content = getattr(document, attr)
                    preview = content[:100] + "..." if len(content) > 100 else content
                    print(f"  {attr}: {preview}")
                else:
                    print(f"  {attr}: [Attribute not found]")
            except Exception as e:
                print(f"  {attr}: [Error: {e}]")
        
        # Check metadata
        print("\nDocument metadata:")
        for attr in ["title", "author", "filename", "hash", "path", "processed_time"]:
            try:
                if hasattr(document, attr):
                    print(f"  {attr}: {getattr(document, attr)}")
                else:
                    print(f"  {attr}: [Attribute not found]")
            except Exception as e:
                print(f"  {attr}: [Error: {e}]")
                
        print("\nSuggested attribute to use for text content:")
        if hasattr(document, "content"):
            print("  document.content")
        elif hasattr(document, "extracted_text"):
            print("  document.extracted_text")
        elif hasattr(document, "text"):
            print("  document.text")
        elif hasattr(document, "fulltext"):
            print("  document.fulltext")
        else:
            print("  [No text content attribute found]")
        
    except Exception as e:
        print(f"Error examining document: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_document.py path/to/pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    check_document(pdf_path)
