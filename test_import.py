# test_import.py
try:
    from retriever import Retriever
    print("Successfully imported Retriever!")
except ImportError as e:
    print(f"Import failed: {str(e)}")