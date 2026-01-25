"""
RAG integration module for generating evidence-backed explanations.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rag_framework.rag_model_chroma import load_rag_model
from src.config import IDX_TO_CLASS, RAG_TOP_K, RAG_USE_CHROMA, RAG_CHROMA_PATH, RAG_USE_LLM, RAG_LLM_MODEL

# Cell type mapping for better query construction
CELL_TYPE_MAPPING = {
    'BLA': ['blast cells', 'blasts', 'blast', 'immature cells', 'blast cell morphology'],
    'EOS': ['eosinophils', 'eosinophil', 'eosinophilic', 'eosinophil cells'],
    'LYT': ['lymphocytes', 'lymphocyte', 'lymphoid cells', 'lymphocytic'],
    'MON': ['monocytes', 'monocyte', 'monocytic', 'monocyte cells'],
    'NGS': ['neutrophils', 'neutrophil', 'segmented neutrophils', 'neutrophil cells'],
    'NIF': ['immature neutrophils', 'band cells', 'stab cells', 'neutrophil precursors', 'band neutrophils'],
    'PMO': ['promyelocytes', 'promyelocyte', 'early granulocyte precursors', 'promyelocyte cells']
}


def generate_explanation(predicted_class_name, confidence, uncertainty, additional_context=None):
    """
    Generate RAG-based explanation for a prediction.
    
    Args:
        predicted_class_name: Name of predicted class (e.g., 'BLA', 'EOS')
        confidence: Confidence score (0-1)
        uncertainty: Uncertainty value
        additional_context: Additional context string (optional)
        
    Returns:
        Dictionary with explanation and metadata
    """
    # Load RAG model (now uses Chroma by default)
    rag = None
    if RAG_USE_CHROMA:
        try:
            rag = load_rag_model(
                top_k=RAG_TOP_K,
                use_chroma=True,
                chroma_path=RAG_CHROMA_PATH,
                use_llm=RAG_USE_LLM,
                llm_model=RAG_LLM_MODEL
            )
        except (ValueError, Exception) as e:
            # Fallback to CSV model if Chroma fails
            import warnings
            warnings.warn(f"Chroma database failed, falling back to CSV model: {e}")
            rag = load_rag_model(
                top_k=RAG_TOP_K,
                use_chroma=False,  # Use CSV fallback
                use_llm=RAG_USE_LLM,
                llm_model=RAG_LLM_MODEL
            )
    else:
        rag = load_rag_model(
            top_k=RAG_TOP_K,
            use_chroma=False,
            use_llm=RAG_USE_LLM,
            llm_model=RAG_LLM_MODEL
        )
    
    # Build improved query with cell type expansion
    confidence_pct = confidence * 100
    
    # Get cell type terms from mapping
    cell_terms = CELL_TYPE_MAPPING.get(predicted_class_name, [predicted_class_name])
    primary_term = cell_terms[0]  # Use most common term
    
    # Build comprehensive query
    query_parts = [
        f"{primary_term}",
        f"bone marrow",
        f"morphology characteristics",
        f"clinical features",
        f"diagnosis significance"
    ]
    
    # Add alternative terms for better matching (limit to avoid too long queries)
    if len(cell_terms) > 1:
        query_parts.extend(cell_terms[1:3])  # Add 2 more alternative terms
    
    if additional_context:
        query_parts.append(additional_context)
    
    query = " ".join(query_parts)
    
    # Get RAG explanation
    result = rag.query(query)
    
    # Format explanation
    explanation = result['explanation']
    
    # Add confidence and uncertainty context
    explanation_header = f"**Predicted Class: {predicted_class_name}**\n\n"
    explanation_header += f"**Confidence: {confidence_pct:.1f}%** | **Uncertainty: {uncertainty:.4f}**\n\n"
    explanation_header += "**Clinical Explanation:**\n\n"
    
    full_explanation = explanation_header + explanation
    
    return {
        'predicted_class': predicted_class_name,
        'confidence': confidence,
        'uncertainty': uncertainty,
        'explanation': full_explanation,
        'raw_explanation': explanation,
        'retrieved_documents': result['retrieved_documents'],
        'num_sources': result['num_sources']
    }


def generate_explanation_from_prediction(prediction_result, uncertainty_result=None):
    """
    Generate explanation from prediction and uncertainty results.
    
    Args:
        prediction_result: Dictionary from predict_single_image
        uncertainty_result: Dictionary from estimate_uncertainty_for_single (optional)
        
    Returns:
        Dictionary with explanation
    """
    predicted_class = prediction_result['predicted_class']
    confidence = prediction_result['confidence']
    
    # Get uncertainty if provided
    if uncertainty_result is not None:
        uncertainty = uncertainty_result.get('total_uncertainty', 0.0)
    else:
        uncertainty = 1.0 - confidence  # Approximate uncertainty
    
    return generate_explanation(predicted_class, confidence, uncertainty)


if __name__ == "__main__":
    # Test RAG integration
    print("Testing RAG integration...")
    
    # Simulate a prediction
    prediction = {
        'predicted_class': 'BLA',
        'confidence': 0.95
    }
    
    uncertainty = {
        'total_uncertainty': 0.05
    }
    
    explanation = generate_explanation_from_prediction(prediction, uncertainty)
    
    print("\nGenerated Explanation:")
    print("=" * 60)
    print(explanation['explanation'])


