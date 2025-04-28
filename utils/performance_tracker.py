#!/usr/bin/env python3
# performance_tracker.py - Utility for tracking performance metrics

"""
This module provides utilities for tracking performance metrics:
- PDF processing metrics (tokens processed, time, tokens/sec)
- Embedding generation metrics
- LLM query performance (tokens generated, generation speed)
- Memory usage

Usage:
  from utils.performance_tracker import PerformanceTracker, track_function
"""

import time
import functools
import json
import os
import psutil
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Callable, List, Optional, Union

# Constants
METRICS_FOLDER = "metrics"
METRICS_FILE = f"{METRICS_FOLDER}/performance_metrics.json"

@dataclass
class PDFProcessingMetrics:
    """Metrics for PDF processing operations"""
    pdf_path: str
    pdf_size_bytes: int
    extracted_text_length: int
    num_tokens: int
    num_chunks: int
    processing_time_seconds: float
    tokens_per_second: float
    embedding_time_seconds: float
    peak_memory_mb: float
    timestamp: str

    #configuration tracking fields
    chunk_size: int
    tokenizer_name: str
    embedding_model: str

@dataclass
class QueryMetrics:
    """Metrics for query operations"""
    query: str
    query_length: int
    retrieval_time_seconds: float
    llm_model: str
    llm_generation_time_seconds: float
    response_length: int
    num_chunks_retrieved: int
    peak_memory_mb: float
    timestamp: str

class PerformanceTracker:
    """Class for tracking various performance metrics"""
    
    def __init__(self):
        """Initialize the performance tracker"""
        os.makedirs(METRICS_FOLDER, exist_ok=True)
        self.process = psutil.Process(os.getpid())
        self._load_existing_metrics()
    
    def _load_existing_metrics(self):
        """Load existing metrics from file if it exists"""
        self.metrics = {
            "pdf_processing": [],
            "queries": []
        }
        
        if os.path.exists(METRICS_FILE):
            try:
                with open(METRICS_FILE, 'r') as f:
                    file_contents = f.read().strip()
                    # If file is empty, use default metrics
                    if not file_contents:
                        print(f"Warning: Metrics file is empty. Using default metrics.")
                        return
                    
                    try:
                        loaded_metrics = json.loads(file_contents)
                    except json.JSONDecodeError as json_err:
                        print(f"Warning: Invalid JSON in metrics file ({json_err}). Using default metrics.")
                        return
                    
                    # Validate the loaded metrics structure
                    if not isinstance(loaded_metrics, dict):
                        print(f"Warning: Invalid metrics file format. Using default metrics.")
                        return
                    
                    # Ensure both keys exist, use default if not
                    self.metrics["pdf_processing"] = loaded_metrics.get("pdf_processing", [])
                    self.metrics["queries"] = loaded_metrics.get("queries", [])
                    
            except IOError as e:
                print(f"Warning: Could not read metrics file: {e}. Using default metrics.")
    
    def _save_metrics(self):
        """Save metrics to disk"""
        try:
            os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
            with open(METRICS_FILE, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except IOError as e:
            print(f"Error saving metrics: {e}")
    
    def get_peak_memory_mb(self) -> float:
        """Get peak memory usage in MB"""
        return self.process.memory_info().rss / (1024 * 1024)
    
    def record_pdf_processing(self, metrics: PDFProcessingMetrics):
        """Record PDF processing metrics"""
        self.metrics["pdf_processing"].append(asdict(metrics))
        self._save_metrics()
    
    def record_query(self, metrics: QueryMetrics):
        """Record query metrics"""
        self.metrics["queries"].append(asdict(metrics))
        self._save_metrics()
    
    def get_pdf_processing_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics for PDF processing"""
        if not self.metrics["pdf_processing"]:
            return {"error": "No PDF processing metrics recorded"}
        
        total_pdfs = len(self.metrics["pdf_processing"])
        total_tokens = sum(m["num_tokens"] for m in self.metrics["pdf_processing"])
        total_time = sum(m["processing_time_seconds"] for m in self.metrics["pdf_processing"])
        avg_tokens_per_second = sum(m["tokens_per_second"] for m in self.metrics["pdf_processing"]) / total_pdfs
        
        return {
            "total_pdfs_processed": total_pdfs,
            "total_tokens_processed": total_tokens,
            "total_processing_time_seconds": total_time,
            "average_tokens_per_second": avg_tokens_per_second,
            "average_chunks_per_pdf": sum(m["num_chunks"] for m in self.metrics["pdf_processing"]) / total_pdfs
        }
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics for queries"""
        if not self.metrics["queries"]:
            return {"error": "No query metrics recorded"}
        
        total_queries = len(self.metrics["queries"])
        avg_retrieval_time = sum(m["retrieval_time_seconds"] for m in self.metrics["queries"]) / total_queries
        avg_llm_time = sum(m["llm_generation_time_seconds"] for m in self.metrics["queries"]) / total_queries
        
        # Group by model
        model_stats = {}
        for m in self.metrics["queries"]:
            model = m["llm_model"]
            if model not in model_stats:
                model_stats[model] = {
                    "count": 0,
                    "total_time": 0,
                    "total_response_length": 0
                }
            model_stats[model]["count"] += 1
            model_stats[model]["total_time"] += m["llm_generation_time_seconds"]
            model_stats[model]["total_response_length"] += m["response_length"]
        
        # Calculate per-model averages
        for model, stats in model_stats.items():
            stats["avg_time"] = stats["total_time"] / stats["count"]
            stats["avg_response_length"] = stats["total_response_length"] / stats["count"]
            stats["chars_per_second"] = stats["total_response_length"] / stats["total_time"] if stats["total_time"] > 0 else 0
        
        return {
            "total_queries": total_queries,
            "average_retrieval_time": avg_retrieval_time,
            "average_llm_generation_time": avg_llm_time,
            "models": model_stats
        }

def track_function(category: str):
    """Decorator to track function execution time"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            print(f"[{category}] {func.__name__} executed in {execution_time:.2f} seconds")
            return result
        return wrapper
    return decorator

# Create a global instance for easy import
performance_tracker = PerformanceTracker()