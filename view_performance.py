#!/usr/bin/env python3
# view_performance.py - Display performance metrics

"""
This script displays performance metrics for PDF processing and query operations.

Usage:
  python3 view_performance.py [--pdf | --query | --all]

Arguments:
  --pdf       Show only PDF processing metrics (default if no flag specified)
  --query     Show only query metrics
  --all       Show all metrics
"""

import os
import sys
import argparse
import json
from datetime import datetime
from tabulate import tabulate

# Try to import performance tracking utilities
try:
    from utils.performance_tracker import performance_tracker
except ImportError:
    print("Error: Performance tracking utilities not found.")
    print("Please ensure the performance_tracker.py file exists in the utils directory.")
    sys.exit(1)

def display_pdf_processing_metrics():
    """Display PDF processing metrics in a nice table"""
    stats = performance_tracker.get_pdf_processing_stats()
    
    # Display summary
    print("\n===== PDF Processing Summary =====")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Display detailed metrics for each PDF if available
    if performance_tracker.metrics["pdf_processing"]:
        print("\n===== Individual PDF Processing Metrics =====")
        
        # Prepare table data with shorter, more concise headers
        headers = [
            "#", 
            "Size(MB)", 
            "Text Len", 
            "Tokens", 
            "Chunks", 
            "Total Time(s)", 
            "Tokens/sec", 
            "Mem(MB)", 
            "Date", 
            "Chunk", 
            "Tokenizer", 
            "Embedding"
        ]
        rows = []
        
        for idx, entry in enumerate(performance_tracker.metrics["pdf_processing"], 1):
            # Calculate total processing time
            total_processing_time = entry['processing_time_seconds'] + entry.get('embedding_time_seconds', 0)
            
            # Parse and format timestamp
            timestamp = datetime.fromisoformat(entry["timestamp"])
            formatted_date = timestamp.strftime("%Y-%m-%d")
            
            row = [
                idx,  # Replace PDF name with a number
                f"{entry['pdf_size_bytes'] / (1024 * 1024):.2f}",
                entry["extracted_text_length"],
                entry["num_tokens"],
                entry["num_chunks"],
                f"{total_processing_time:.2f}",
                f"{entry['tokens_per_second']:.2f}",
                f"{entry['peak_memory_mb']:.2f}",
                formatted_date,
                entry.get("chunk_size", "N/A"),
                entry.get("tokenizer_name", "N/A"),
                entry.get("embedding_model", "N/A")
            ]
            rows.append(row)
        
        print(tabulate(rows, headers=headers, tablefmt="simple"))
    else:
        print("\nNo PDF processing metrics recorded yet.")

def display_query_metrics():
    """Display query metrics in a nice table"""
    stats = performance_tracker.get_query_stats()
    
    # Display summary
    print("\n===== Query Summary =====")
    for key, value in stats.items():
        if key != "models":
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
    
    # Display model-specific stats
    if "models" in stats:
        print("\n===== Performance by Model =====")
        model_headers = ["Model", "Count", "Avg Time", "Avg Response Length", "Chars/sec"]
        model_rows = []
        
        for model, model_stats in stats["models"].items():
            model_rows.append([
                model,
                model_stats["count"],
                f"{model_stats['avg_time']:.2f}s",
                model_stats["avg_response_length"],
                f"{model_stats['chars_per_second']:.2f}"
            ])
        
        print(tabulate(model_rows, headers=model_headers, tablefmt="grid"))
    
    # Display detailed query metrics
    if performance_tracker.metrics["queries"]:
        print("\n===== Recent Query Metrics =====")
        
        # Limit to 10 most recent queries for readability
        recent_queries = performance_tracker.metrics["queries"][-10:]
        
        # Prepare table data
        headers = ["Query", "Model", "Retrieval Time", "Generation Time", "Response Length"]
        rows = []
        
        for entry in recent_queries:
            # Truncate query if too long
            query = entry["query"]
            if len(query) > 50:
                query = query[:47] + "..."
            
            rows.append([
                query,
                entry["llm_model"],
                f"{entry['retrieval_time_seconds']:.2f}s",
                f"{entry['llm_generation_time_seconds']:.2f}s",
                entry["response_length"]
            ])
        
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        print("\nNo query metrics recorded yet.")

def main():
    parser = argparse.ArgumentParser(description="Display performance metrics")
    parser.add_argument("--pdf", action="store_true", help="Show PDF processing metrics")
    parser.add_argument("--query", action="store_true", help="Show query metrics")
    parser.add_argument("--all", action="store_true", help="Show all metrics")
    
    args = parser.parse_args()
    
    # If no specific flag, show PDF metrics by default
    if not (args.pdf or args.query or args.all):
        args.pdf = True
    
    # Show metrics based on flags
    if args.pdf or args.all:
        display_pdf_processing_metrics()
    
    if args.query or args.all:
        display_query_metrics()

if __name__ == "__main__":
    main()