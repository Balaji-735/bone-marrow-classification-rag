"""
Simple test for RAG integration without requiring torch.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Mock config values to avoid torch dependency
sys.modules['src.config'] = type(sys)('config')
import src.config as config
config.RAG_TOP_K = 3
config.RAG_USE_CHROMA = True
config.RAG_CHROMA_PATH = None
config.RAG_USE_LLM = False
config.RAG_LLM_MODEL = "mistral"

def test_rag_integration():
    """Test RAG integration."""
    print("=" * 60)
    print("Testing RAG Integration (Chroma-based)")
    print("=" * 60)
    
    try:
        # Import after mocking config
        from src.rag_integration import generate_explanation
        
        print("\n1. Testing generate_explanation function...")
        
        # Simulate a prediction
        predicted_class = 'BLA'
        confidence = 0.95
        uncertainty = 0.05
        
        result = generate_explanation(
            predicted_class_name=predicted_class,
            confidence=confidence,
            uncertainty=uncertainty
        )
        
        print(f"\n   Predicted Class: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Uncertainty: {result['uncertainty']}")
        print(f"   Number of Sources: {result['num_sources']}")
        print(f"\n   Explanation (first 500 chars):")
        print(f"   {result['explanation'][:500]}...")
        
        print("\n   [OK] RAG integration test successful!")
        return True
        
    except Exception as e:
        print(f"\n   [ERROR] RAG integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rag_integration()
    sys.exit(0 if success else 1)






