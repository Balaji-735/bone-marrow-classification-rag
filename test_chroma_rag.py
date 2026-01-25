"""
Test script for Chroma RAG integration with fallback to CSV model.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_chroma_rag():
    """Test Chroma RAG with fallback to CSV model."""
    print("=" * 60)
    print("Testing Chroma RAG Integration")
    print("=" * 60)
    
    try:
        from rag_framework.rag_model_chroma import load_rag_model
        
        print("\n1. Attempting to load Chroma RAG model...")
        rag = load_rag_model(use_chroma=True, top_k=3, use_llm=False)
        print("   [OK] Chroma RAG model loaded successfully!")
        
        print("\n2. Testing query...")
        test_query = "Explain the clinical significance of BLA cells in bone marrow"
        result = rag.query(test_query)
        
        print(f"\n   Query: {result['query']}")
        print(f"\n   Retrieved {result['num_sources']} documents")
        print(f"\n   Explanation (first 300 chars):")
        print(f"   {result['explanation'][:300]}...")
        
        print("\n   [OK] Chroma RAG test successful!")
        return True
        
    except Exception as e:
        print(f"\n   [ERROR] Chroma RAG failed: {e}")
        print("\n3. Falling back to CSV-based RAG model...")
        
        try:
            from rag_framework.rag_model import SimpleRAGModel
            
            rag = SimpleRAGModel(top_k=3)
            print("   [OK] CSV RAG model loaded successfully!")
            
            test_query = "Explain the clinical significance of BLA cells in bone marrow"
            result = rag.query(test_query)
            
            print(f"\n   Query: {result['query']}")
            print(f"\n   Retrieved {result['num_sources']} documents")
            print(f"\n   Explanation (first 300 chars):")
            print(f"   {result['explanation'][:300]}...")
            
            print("\n   [OK] CSV RAG fallback test successful!")
            print("\n   Note: To use Chroma RAG, please:")
            print("   1. Ensure Chroma database is not locked")
            print("   2. Run: cd rag-tutorial-v2-main && python populate_database.py --reset")
            return False
            
        except Exception as e2:
            print(f"\n   [ERROR] CSV RAG fallback also failed: {e2}")
            return False

if __name__ == "__main__":
    success = test_chroma_rag()
    sys.exit(0 if success else 1)

