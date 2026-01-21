"""
Streamlit dashboard for bone marrow cell classification.
"""

import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import CLASSES, IDX_TO_CLASS, IMAGE_SIZE, CHATBOT_LLM_MODEL
from src.model_training import ViTClassifier
from src.test_inference import load_trained_model, predict_single_image
from src.uncertainty_estimation import estimate_uncertainty_for_single
from src.explainability import visualize_explanations, overlay_heatmap, generate_gradcam, generate_attention_map
from src.rag_integration import generate_explanation_from_prediction
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from src.data_preprocessing import get_transforms
from src.evaluation_metrics import plot_confusion_matrix, plot_roc_curves
import json


# Page configuration
st.set_page_config(
    page_title="Bone Marrow Cell Classification",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load trained model (cached)."""
    try:
        model = load_trained_model()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please train the model first using: `python main.py train`")
        return None


def preprocess_image(uploaded_file):
    """Preprocess uploaded image."""
    try:
        image = Image.open(uploaded_file).convert('RGB')
        transform = get_transforms('test')
        image_tensor = transform(image).unsqueeze(0)
        return image, image_tensor
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<div class="main-header">🔬 Bone Marrow Cell Classification System</div>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Predict", "Model Performance", "About"]
    )
    
    if page == "Predict":
        predict_page()
    elif page == "Model Performance":
        performance_page()
    elif page == "About":
        about_page()


def generate_template_response(question, rag_context, predicted_class, retrieved_docs):
    """
    Generate a template-based response from RAG context when LLM is unavailable.
    Uses CSV knowledge base as primary source.
    """
    question_lower = question.lower()
    
    # First, try to get info from CSV knowledge base
    try:
        from rag_framework.knowledge_base.hematology_knowledge import get_knowledge_base
        kb = get_knowledge_base()
        cell_info = kb[kb['title'].str.contains(predicted_class, case=False, na=False)]
        
        if not cell_info.empty:
            info = cell_info.iloc[0]
            base_info = info['content']
            source = info['source']
        else:
            base_info = None
            source = None
    except:
        base_info = None
        source = None
    
    # Simple keyword matching to provide relevant information
    response_parts = []
    
    # Check for cell type questions
    if "what is" in question_lower or "what are" in question_lower:
        if base_info:
            response_parts.append(f"**{predicted_class}** cells:\n\n{base_info}")
            if source:
                response_parts.append(f"\n\n*Source: {source}*")
        else:
            # Cell-specific default answers
            cell_defaults = {
                'BLA': 'Blast cells are immature blood cells that are precursors to mature blood cells. In healthy bone marrow, blasts typically account for less than 5% of cells. Elevated blast counts (>20%) are a hallmark of acute leukemia. Blast cells have large nuclei, prominent nucleoli, and minimal cytoplasm.',
                'EOS': 'Eosinophils are granulocytes that play a key role in allergic reactions and defense against parasites. They contain distinctive red-orange granules when stained with Wright-Giemsa. Normal eosinophil count in bone marrow is 1-3%.',
                'LYT': 'Lymphocytes are white blood cells involved in immune responses. They include B cells, T cells, and NK cells. In bone marrow, lymphocytes typically represent 10-20% of cells.',
                'MON': 'Monocytes are large white blood cells that differentiate into macrophages and dendritic cells. They have kidney-shaped or lobulated nuclei and abundant gray-blue cytoplasm. Normal monocyte count in bone marrow is 2-8%.',
                'NGS': 'Neutrophils are the most abundant white blood cells, comprising 50-70% of circulating leukocytes. They have segmented nuclei (2-5 lobes) and fine pink-purple granules.',
                'NIF': 'Immature neutrophils, also called band cells or stab cells, are precursors to mature segmented neutrophils. They have horseshoe or U-shaped nuclei and are typically less than 5% of neutrophils.',
                'PMO': 'Promyelocytes are early granulocyte precursors found in bone marrow. They are larger than myelocytes and have round to oval nuclei with prominent nucleoli.'
            }
            default = cell_defaults.get(predicted_class, f"{predicted_class} cells are a type of bone marrow cell.")
            response_parts.append(f"**{predicted_class}** cells:\n\n{default}")
    
    # Check for clinical significance questions
    elif "clinical" in question_lower or "significance" in question_lower or "meaning" in question_lower:
        if base_info:
            response_parts.append(f"**Clinical Significance of {predicted_class} cells:**\n\n{base_info}")
            if source:
                response_parts.append(f"\n\n*Source: {source}*")
        else:
            response_parts.append(f"The clinical significance of **{predicted_class}** cells:\n\n")
            response_parts.append("Please refer to the RAG explanation above for detailed clinical information.")
    
    # Check for morphology questions
    elif "morphology" in question_lower or "appearance" in question_lower or "look like" in question_lower:
        if base_info:
            # Extract morphology-related sentences
            sentences = base_info.split('.')
            morph_sentences = [s.strip() for s in sentences if any(term in s.lower() for term in ['nucleus', 'cytoplasm', 'granule', 'shape', 'size', 'morphology', 'nuclei'])]
            if morph_sentences:
                response_parts.append(f"**Morphological characteristics of {predicted_class} cells:**\n\n")
                response_parts.extend([f"• {s}." for s in morph_sentences[:3]])
            else:
                response_parts.append(f"**{predicted_class}** cells:\n\n{base_info}")
            if source:
                response_parts.append(f"\n\n*Source: {source}*")
        else:
            response_parts.append(f"Morphological characteristics of **{predicted_class}** cells:\n\n")
            response_parts.append("Please refer to the RAG explanation above for detailed morphological information.")
    
    # Default response
    else:
        if base_info:
            response_parts.append(f"**{predicted_class}** cells:\n\n{base_info}")
            if source:
                response_parts.append(f"\n\n*Source: {source}*")
        elif retrieved_docs:
            # Use first relevant document snippet
            first_doc = retrieved_docs[0]
            content = first_doc.get('content', '')
            # Extract first meaningful sentence
            sentences = content.split('.')
            for sentence in sentences:
                if len(sentence.strip()) > 30:
                    response_parts.append(sentence.strip() + ".")
                    break
    
    return "\n".join(response_parts) if response_parts else f"Please refer to the RAG explanation above for information about {predicted_class} cells."


def predict_page():
    """Prediction page."""
    st.header("🔍 Cell Classification & Explanation")
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a bone marrow cell image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image of a bone marrow cell for classification"
    )
    
    if uploaded_file is not None:
        # Reset chat history for new image
        if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.chat_history = []
            st.session_state.last_uploaded_file = uploaded_file.name
        
        # Preprocess image
        original_image, image_tensor = preprocess_image(uploaded_file)
        
        if original_image is not None and image_tensor is not None:
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                st.image(original_image, use_container_width=True)
            
            # Run prediction
            with st.spinner("Analyzing image..."):
                # Get prediction
                prediction = predict_single_image(model, image_tensor)
                
                # Get uncertainty
                uncertainty_result = estimate_uncertainty_for_single(model, image_tensor)
                
                # Get RAG explanation
                explanation_result = generate_explanation_from_prediction(
                    prediction, uncertainty_result
                )
                
                # Extract variables for display and chatbot
                predicted_class = prediction['predicted_class']
                confidence = prediction['confidence']
                confidence_pct = confidence * 100
                uncertainty = uncertainty_result['total_uncertainty']
            
            # Display results
            with col2:
                st.subheader("Prediction Results")
                
                # Metrics
                st.metric("Predicted Class", predicted_class)
                st.metric("Confidence", f"{confidence_pct:.2f}%")
                st.metric("Uncertainty", f"{uncertainty:.4f}")
                
                # Confidence indicator
                if confidence > 0.9:
                    st.success("✅ High Confidence Prediction")
                elif confidence > 0.7:
                    st.warning("⚠️ Moderate Confidence Prediction")
                else:
                    st.error("❌ Low Confidence Prediction - Review Recommended")
            
            # Probability distribution
            st.subheader("Class Probabilities")
            prob_data = prediction['probabilities']
            classes = list(prob_data.keys())
            probs = list(prob_data.values())
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(classes, probs, color='steelblue', alpha=0.7)
            ax.set_xlabel('Probability', fontsize=12)
            ax.set_ylabel('Class', fontsize=12)
            ax.set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
            ax.set_xlim([0, 1])
            
            # Highlight predicted class
            pred_idx = classes.index(predicted_class)
            bars[pred_idx].set_color('red')
            
            # Add probability labels
            for i, (bar, prob) in enumerate(zip(bars, probs)):
                ax.text(prob + 0.01, i, f'{prob:.3f}', 
                       va='center', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Explanations section
            st.subheader("🔬 Visual Explanations")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("**Grad-CAM Heatmap**")
                try:
                    cam = generate_gradcam(model, image_tensor)
                    overlaid = overlay_heatmap(original_image, cam)
                    st.image(overlaid, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate Grad-CAM: {e}")
            
            with col4:
                st.markdown("**ViT Attention Map**")
                try:
                    attn_map = generate_attention_map(model, image_tensor)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(attn_map, cmap='hot', interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title('Attention Visualization', fontsize=12)
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Could not generate attention map: {e}")
            
            # RAG Explanation
            st.subheader("📚 Clinical Explanation (RAG-Generated)")
            st.markdown(explanation_result['explanation'])
            
            # Show retrieved sources
            with st.expander("View Retrieved Sources"):
                for doc in explanation_result['retrieved_documents']:
                    st.markdown(f"**{doc['title']}**")
                    st.markdown(f"*Source: {doc['source']}*")
                    st.markdown(f"Relevance Score: {doc['score']:.3f}")
                    st.markdown("---")
            
            # Chatbot section
            st.subheader("💬 Ask Questions About This Prediction")
            st.markdown("Ask questions about the predicted cell type, its clinical significance, or related information.")
            
            # Initialize chat history in session state
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Prepare RAG context for chatbot
            rag_context = "\n\n---\n\n".join([
                f"Source: {doc.get('source', 'Unknown')} (Page {doc.get('page', 'N/A')})\n{doc.get('content', '')}"
                for doc in explanation_result['retrieved_documents']
            ])
            
            # System prompt for chatbot
            chatbot_prompt_template = """You are a helpful medical assistant specializing in hematology and bone marrow cell classification. 
Answer questions based on the following context from research papers about bone marrow cells.

Context from retrieved documents:
{context}

Current Prediction Information:
- Predicted Cell Type: {predicted_class}
- Confidence: {confidence}%
- Uncertainty: {uncertainty}

Answer the user's question based on the context provided. If the context doesn't contain enough information, say so clearly. 
Be concise, accurate, and focus on clinical and morphological information.

User Question: {question}
Assistant:"""
            
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            user_question = st.chat_input("Ask a question about this prediction...")
            
            if user_question:
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(user_question)
                
                # Generate response with Ollama
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            # Try to get response from Ollama with fallback
                            response = None
                            models_to_try = [CHATBOT_LLM_MODEL, "gemma:2b"]
                            
                            for model_name in models_to_try:
                                try:
                                    # Create prompt
                                    prompt_template = ChatPromptTemplate.from_template(chatbot_prompt_template)
                                    prompt = prompt_template.format(
                                        context=rag_context,
                                        predicted_class=predicted_class,
                                        confidence=confidence_pct,
                                        uncertainty=uncertainty,
                                        question=user_question
                                    )
                                    
                                    llm = Ollama(model=model_name, timeout=30)
                                    response = llm.invoke(prompt)
                                    break  # Success, exit loop
                                except Exception as model_error:
                                    error_str = str(model_error).lower()
                                    if "memory" in error_str or "500" in error_str or "insufficient" in error_str or "cuda" in error_str:
                                        # Try next model if memory issue
                                        continue
                                    else:
                                        # For other errors, try next model once more
                                        if model_name == models_to_try[-1]:
                                            continue  # Don't raise, will use fallback
                                        continue
                            
                            if not response:
                                # Fallback: Use template-based response from RAG context
                                response = generate_template_response(
                                    user_question, 
                                    rag_context, 
                                    predicted_class, 
                                    explanation_result['retrieved_documents']
                                )
                            
                            # Display response
                            st.markdown(response)
                            # Add assistant response to chat history
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            
                        except Exception as e:
                            # Final fallback - use template response
                            try:
                                response = generate_template_response(
                                    user_question, 
                                    rag_context, 
                                    predicted_class, 
                                    explanation_result['retrieved_documents']
                                )
                                st.markdown(response)
                                st.session_state.chat_history.append({"role": "assistant", "content": response})
                            except Exception as fallback_error:
                                # Last resort - provide basic info
                                from rag_framework.knowledge_base.hematology_knowledge import get_knowledge_base
                                kb = get_knowledge_base()
                                cell_info = kb[kb['title'].str.contains(predicted_class, case=False, na=False)]
                                
                                if not cell_info.empty:
                                    info = cell_info.iloc[0]
                                    response = f"**{info['title']}**\n\n{info['content']}\n\n*Source: {info['source']}*"
                                else:
                                    response = (
                                        f"**{predicted_class}** cells are a type of bone marrow cell. "
                                        f"Please refer to the RAG explanation above for detailed information. "
                                        f"\n\n*Note: LLM responses are unavailable due to insufficient system memory.*"
                                    )
                                
                                st.info(response)
                                st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Clear chat button
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()


def performance_page():
    """Model performance page."""
    st.header("📊 Model Performance Metrics")
    
    # Load metrics if available
    metrics_file = Path("results/metrics/classification_metrics.json")
    
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Overall accuracy
        st.subheader("Overall Performance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Test Accuracy", f"{metrics['accuracy']*100:.2f}%")
        
        if 'mean_roc_auc' in metrics:
            with col2:
                st.metric("Mean ROC AUC", f"{metrics['mean_roc_auc']:.4f}")
        
        # Classification report
        st.subheader("Classification Report")
        report = metrics['classification_report']
        
        # Create DataFrame for better display
        import pandas as pd
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        # Visualizations
        st.subheader("Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Confusion Matrix**")
            confusion_matrix_path = Path("results/visualizations/confusion_matrix.png")
            if confusion_matrix_path.exists():
                st.image(str(confusion_matrix_path), use_container_width=True)
            else:
                st.info("Confusion matrix not available. Run evaluation first.")
        
        with col2:
            st.markdown("**ROC Curves**")
            roc_path = Path("results/visualizations/roc_curves.png")
            if roc_path.exists():
                st.image(str(roc_path), use_container_width=True)
            else:
                st.info("ROC curves not available. Run evaluation first.")
        
        # Class distribution
        class_dist_path = Path("results/visualizations/class_distribution.png")
        if class_dist_path.exists():
            st.markdown("**Class Distribution**")
            st.image(str(class_dist_path), use_container_width=True)
    
    else:
        st.info("Performance metrics not available. Please run evaluation first using: `python main.py eval`")


def about_page():
    """About page."""
    st.header("About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This system implements an **Explainable Deep Learning** approach for multi-class bone marrow cell type classification 
    with RAG-based evidence explanations. The project supports the **UN Sustainable Development Goal 3: Good Health and Well-being** 
    by providing AI-assisted diagnostic support for hematologic analysis.
    
    ### 🔬 Key Features
    
    - **Vision Transformer (ViT) Classifier**: State-of-the-art deep learning model for cell classification
    - **7 Cell Types**: BLA (Blast), EOS (Eosinophil), LYT (Lymphocyte), MON (Monocyte), NGS (Neutrophil), NIF (Immature Neutrophil), PMO (Promyelocyte)
    - **Explainable AI (XAI)**:
      - Grad-CAM heatmaps showing which image regions influence predictions
      - ViT attention maps visualizing model focus areas
    - **Uncertainty Estimation**: Monte Carlo Dropout for quantifying prediction confidence
    - **RAG Integration**: Retrieval-Augmented Generation for evidence-backed clinical explanations
    - **Classical ML Baselines**: SVM, Random Forest, and XGBoost for comparison
    
    ### 🧠 Technology Stack
    
    - **Deep Learning**: PyTorch, Vision Transformer (timm)
    - **Explainability**: Grad-CAM, Attention Visualization
    - **Uncertainty**: Monte Carlo Dropout
    - **RAG**: Knowledge base retrieval with templated generation
    - **Dashboard**: Streamlit
    - **Evaluation**: scikit-learn metrics, ROC curves, confusion matrices
    
    ### 📚 Dataset
    
    The model is trained on the [Bone Marrow Cell Classification Dataset](https://www.kaggle.com/datasets/donajui/bone-marrow-cell-classification) 
    from Kaggle, containing images of 7 different bone marrow cell types.
    
    ### 🚀 Usage
    
    1. **Train the model**: `python main.py train`
    2. **Evaluate**: `python main.py eval`
    3. **Run baselines**: `python main.py baselines`
    4. **Launch dashboard**: `streamlit run dashboard/app.py`
    
    ### ⚠️ Medical Disclaimer
    
    This system is designed for research and educational purposes. It should not be used as the sole basis for 
    clinical diagnosis. All predictions should be reviewed by qualified medical professionals.
    
    ### 📖 References
    
    - Vision Transformer: Dosovitskiy et al., "An Image is Worth 16x16 Words", NeurIPS 2020
    - Grad-CAM: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017
    - RAG: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS 2020
    """)


if __name__ == "__main__":
    main()


