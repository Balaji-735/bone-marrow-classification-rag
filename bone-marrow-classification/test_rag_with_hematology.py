"""
Test RAG with hematology PDFs - check what's actually being retrieved.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from rag_framework.rag_model_chroma import ChromaRAGModel

def test_hematology_queries():
    """Test various queries related to bone marrow cells."""
    print("=" * 60)
    print("Testing RAG with Hematology PDFs")
    print("=" * 60)
    
    try:
        rag = ChromaRAGModel(top_k=5, use_llm=False)
        print("\n[OK] Chroma RAG model loaded successfully!")
        
        # Test queries for different cell types
        test_queries = [
            "BLA blast cells bone marrow",
            "eosinophils EOS bone marrow",
            "lymphocytes LYT classification",
            "monocytes MON characteristics",
            "neutrophils NGS morphology",
            "immature neutrophils NIF",
            "promyelocytes PMO"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")
            
            result = rag.query(query)
            
            print(f"\nRetrieved {result['num_sources']} documents")
            
            if result['retrieved_documents']:
                print("\nTop Retrieved Documents:")
                for i, doc in enumerate(result['retrieved_documents'][:3], 1):
                    print(f"\n[{i}] Score: {doc['score']:.4f}")
                    print(f"    Source: {doc['source']}")
                    if doc.get('page') is not None:
                        print(f"    Page: {doc['page']}")
                    print(f"    Content preview: {doc['content'][:200]}...")
            else:
                print("No documents retrieved!")
            
            if result['explanation']:
                print(f"\nExplanation preview: {result['explanation'][:300]}...")
            else:
                print("\nNo explanation generated (empty result)")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hematology_queries()
    sys.exit(0 if success else 1)






