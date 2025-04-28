#!/usr/bin/env python3
# query_processor.py - Process and enhance user queries

"""
This module handles query processing:
1. Preprocesses user queries for better retrieval
2. Extracts key terms from queries
3. Enhances queries for academic/technical contexts
4. Integrates with the performance tracking system

Usage:
  from query_processor import QueryProcessor
  
  processor = QueryProcessor()
  enhanced_query = processor.process_query("What is the impact of X on Y?")
"""

import re
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Import performance tracking
try:
    from utils.performance_tracker import track_function
except ImportError:
    print("Warning: Performance tracking not available")
    # Create dummy decorator if tracking is not available
    def track_function(category):
        def decorator(func):
            return func
        return decorator

@dataclass
class ProcessedQuery:
    """Processed query with metadata"""
    original_query: str
    processed_text: str
    query_type: str  # 'factual', 'conceptual', etc.
    key_terms: List[str]
    processing_time: float

class QueryProcessor:
    """Process and enhance user queries"""
    
    def __init__(self):
        """Initialize the query processor"""
        # Simple patterns for academic queries
        self.definition_pattern = re.compile(r'(?:what|define|explain) (?:is|are) (.+?)(?:\?|$)', re.IGNORECASE)
        self.comparison_pattern = re.compile(r'(?:compare|difference between|similarities between) (.+?) (?:and|with) (.+?)(?:\?|$)', re.IGNORECASE)
        self.impact_pattern = re.compile(r'(?:impact|effect|influence) of (.+?) on (.+?)(?:\?|$)', re.IGNORECASE)
        
        # Stop words to filter out
        self.stop_words = set(['a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as', 'is', 'are', 'was', 'were'])
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key technical terms from the query"""
        # Simple approach: split by spaces, filter stop words, keep terms with capital letters or longer terms
        words = query.split()
        key_terms = []
        
        for word in words:
            # Clean the word
            clean_word = word.strip().lower().strip('.,?!;:()"\'')
            
            # Skip stop words and short words
            if clean_word in self.stop_words or len(clean_word) < 4:
                continue
            
            # Check if original word had capital letters (potential technical term)
            if any(c.isupper() for c in word) or len(clean_word) > 5:
                key_terms.append(clean_word)
        
        return key_terms
    
    def _detect_query_type(self, query: str) -> str:
        """Detect the type of query"""
        if self.definition_pattern.search(query):
            return "definition"
        elif self.comparison_pattern.search(query):
            return "comparison"
        elif self.impact_pattern.search(query):
            return "impact"
        elif '?' not in query and len(query.split()) < 5:
            return "keyword"
        else:
            return "general"
    
    def _preprocess_text(self, text: str) -> str:
        """Basic preprocessing of text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @track_function("Query Processing")
    def process_query(self, query: str) -> ProcessedQuery:
        """Process a query for improved retrieval
        
        Args:
            query: The user query
            
        Returns:
            ProcessedQuery object with enhanced query
        """
        start_time = time.time()
        
        # Basic preprocessing
        processed_text = self._preprocess_text(query)
        
        # Detect query type
        query_type = self._detect_query_type(query)
        
        # Extract key terms
        key_terms = self._extract_key_terms(query)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create and return processed query
        return ProcessedQuery(
            original_query=query,
            processed_text=processed_text,
            query_type=query_type,
            key_terms=key_terms,
            processing_time=processing_time
        )
    
    def prepare_for_embedding(self, processed_query: ProcessedQuery) -> str:
        """Format the processed query for embedding model
        
        Args:
            processed_query: The processed query
            
        Returns:
            Formatted query string
        """
        # Format as 'query: text' for better performance with models like E5
        return f"query: {processed_query.processed_text}"

# Example usage
if __name__ == "__main__":
    import sys
    
    processor = QueryProcessor()
    
    # Example queries
    example_queries = [
        "What is the transformer architecture?",
        "Compare LSTM and GRU networks",
        "How does temperature affect LLM output?",
        "Define vector database",
        "FAISS vs Chroma performance",
    ]
    
    for query in example_queries:
        processed = processor.process_query(query)
        
        print(f"\nOriginal: {processed.original_query}")
        print(f"Type: {processed.query_type}")
        print(f"Key terms: {', '.join(processed.key_terms)}")
        print(f"Embedding format: {processor.prepare_for_embedding(processed)}")