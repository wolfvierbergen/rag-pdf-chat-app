#!/usr/bin/env python3
# pdf_processor.py - Extract text and metadata from PDFs

"""
This module handles PDF processing:
1. Extracts text from PDFs
2. Preserves basic metadata (title, author, date)
3. Tracks page numbers and approximate section headings
4. Integrates with the performance tracking system

Usage:
  from pdf_processor import PDFProcessor
  processor = PDFProcessor()
  document = processor.process("path/to/pdf")
"""

import os
import time
import hashlib
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from PyPDF2 import PdfReader
import re

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
class PageContent:
    """Content of a PDF page with its metadata"""
    page_number: int
    text: str
    is_section_header: bool = False
    section_title: Optional[str] = None
    
@dataclass
class DocumentContent:
    """Processed PDF document with its metadata"""
    path: str
    filename: str
    hash: str
    title: Optional[str]
    author: Optional[str]
    num_pages: int
    creation_date: Optional[str]
    file_size_bytes: int
    pages: List[PageContent]
    processed_time: str
    extraction_time_seconds: float
    
    def get_full_text(self) -> str:
        """Get the full text of the document"""
        return "\n\n".join([page.text for page in self.pages])
    
    def __len__(self) -> int:
        """Return the number of pages"""
        return len(self.pages)

class PDFProcessor:
    """Process PDFs and extract text with metadata"""
    
    def __init__(self):
        """Initialize the PDF processor"""
        self._header_pattern = re.compile(r'^(?:\d+(?:\.\d+)*[\.\s]+)?\s*([A-Z][^\n]{5,100})$', re.MULTILINE)
    
    def _get_pdf_hash(self, file_path: str) -> str:
        """Create a hash of the PDF file to uniquely identify it"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _extract_metadata(self, pdf_reader: PdfReader, file_path: str) -> Dict[str, Any]:
        """Extract basic metadata from the PDF"""
        metadata = {}
        info = pdf_reader.metadata
        
        if info:
            metadata["title"] = info.get("/Title", os.path.basename(file_path))
            metadata["author"] = info.get("/Author", "Unknown")
            metadata["creation_date"] = info.get("/CreationDate", "Unknown")
        else:
            metadata["title"] = os.path.basename(file_path)
            metadata["author"] = "Unknown"
            metadata["creation_date"] = "Unknown"
            
        return metadata
    
    def _detect_section_heading(self, text: str) -> Optional[str]:
        """Detect if a text block is likely a section heading"""
        match = self._header_pattern.search(text)
        if match:
            return match.group(1)
        return None
    
    @track_function("PDF Text Extraction")
    def process(self, file_path: str) -> DocumentContent:
        """Process a PDF file and extract content with metadata"""
        start_time = time.time()
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        pdf_hash = self._get_pdf_hash(file_path)
        
        # Extract text and metadata
        reader = PdfReader(file_path)
        metadata = self._extract_metadata(reader, file_path)
        
        # Process each page
        pages = []
        current_section = None
        
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            
            # Check for section heading at beginning of page
            section_heading = self._detect_section_heading(page_text[:200] if page_text else "")
            if section_heading:
                current_section = section_heading
                is_header = True
            else:
                is_header = False
            
            page_content = PageContent(
                page_number=i + 1,
                text=page_text,
                is_section_header=is_header,
                section_title=current_section
            )
            pages.append(page_content)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create document content object
        document = DocumentContent(
            path=file_path,
            filename=os.path.basename(file_path),
            hash=pdf_hash,
            title=metadata["title"],
            author=metadata["author"],
            num_pages=len(reader.pages),
            creation_date=metadata["creation_date"],
            file_size_bytes=file_size,
            pages=pages,
            processed_time=datetime.now().isoformat(),
            extraction_time_seconds=processing_time
        )
        
        return document

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python pdf_processor.py path/to/pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    processor = PDFProcessor()
    
    try:
        document = processor.process(pdf_path)
        print(f"Processed: {document.filename}")
        print(f"Title: {document.title}")
        print(f"Pages: {document.num_pages}")
        print(f"Process time: {document.extraction_time_seconds:.2f} seconds")
    except Exception as e:
        print(f"Error processing PDF: {e}")