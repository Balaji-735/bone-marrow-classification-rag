"""
Test script to verify RAG integration works in dashboard context.
Simulates what the dashboard does when generating explanations.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_dashboard_rag_flow():
    """Test the RAG flow as used in the dashboard."""
    print("=" * 60)
    print("Testing Dashboard RAG Integration Flow")
    print("=" * 60)
    
    try:
        # Simulate what dashboard does
        from src.rag_integration import generate_explanation_from_prediction
        
        # Simulate a prediction result (as from predict_single_image)
        prediction_result = {
            'predicted_class': 'BLA',
            'confidence': 0.95,
            'probabilities': [0.95, 0.02, 0.01, 0.01, 0.005, 0.002, 0.003]
        }
        
        # Simulate uncertainty result (as from estimate_uncertainty_for_single)
        uncertainty_result = {
            'total_uncertainty': 0.05,
            'epistemic_uncertainty': 0.03,
            'aleatoric_uncertainty': 0.02
        }
        
        print("\n1. Simulating prediction...")
        print(f"   Predicted Class: {prediction_result['predicted_class']}")
        print(f"   Confidence: {prediction_result['confidence']*100:.2f}%")
        
        print("\n2. Generating RAG explanation (as dashboard does)...")
        explanation_result = generate_explanation_from_prediction(
            prediction_result, 
            uncertainty_result
        )
        
        print("\n3. Results:")
        print(f"   Predicted Class: {explanation_result['predicted_class']}")
        print(f"   Confidence: {explanation_result['confidence']*100:.2f}%")
        print(f"   Uncertainty: {explanation_result['uncertainty']:.4f}")
        print(f"   Number of Sources: {explanation_result['num_sources']}")
        
        print("\n4. Explanation Preview:")
        print("-" * 60)
        explanation_text = explanation_result['explanation']
        # Show first 500 characters
        print(explanation_text[:500])
        if len(explanation_text) > 500:
            print("...")
        print("-" * 60)
        
        print("\n5. Retrieved Documents:")
        for i, doc in enumerate(explanation_result['retrieved_documents'][:3], 1):
            print(f"\n   [{i}] {doc.get('title', 'Unknown')}")
            print(f"       Source: {doc.get('source', 'Unknown')}")
            if doc.get('page') is not None:
                print(f"       Page: {doc['page']}")
            print(f"       Score: {doc.get('score', 0):.4f}")
        
        print("\n" + "=" * 60)
        print("[OK] Dashboard RAG integration test successful!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Dashboard RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dashboard_rag_flow()
    sys.exit(0 if success else 1)






