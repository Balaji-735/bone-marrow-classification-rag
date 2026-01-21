"""
Hematology knowledge base for RAG explanations.
"""

import pandas as pd
from pathlib import Path

# Import from config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import KNOWLEDGE_BASE_PATH


def get_knowledge_base(knowledge_base_path=None):
    """
    Load or create hematology knowledge base.
    
    Args:
        knowledge_base_path: Path to knowledge base CSV
        
    Returns:
        DataFrame with knowledge base
    """
    if knowledge_base_path is None:
        knowledge_base_path = KNOWLEDGE_BASE_PATH
    
    knowledge_base_path = Path(knowledge_base_path)
    
    # Create knowledge base if it doesn't exist
    if not knowledge_base_path.exists():
        create_knowledge_base(knowledge_base_path)
    
    # Load and return
    return pd.read_csv(knowledge_base_path)


def create_knowledge_base(save_path):
    """
    Create a sample hematology knowledge base.
    
    Args:
        save_path: Path to save knowledge base CSV
    """
    knowledge_data = [
        {
            'id': 'kb_001',
            'title': 'Blast Cells (BLA) in Bone Marrow',
            'content': 'Blast cells are immature blood cells that are precursors to mature blood cells. In healthy bone marrow, blasts typically account for less than 5% of cells. Elevated blast counts (>20%) are a hallmark of acute leukemia. Blast cells have large nuclei, prominent nucleoli, and minimal cytoplasm. They are critical for diagnosing acute myeloid leukemia (AML) and acute lymphoblastic leukemia (ALL).',
            'source': 'Hematology Textbook, 2023'
        },
        {
            'id': 'kb_002',
            'title': 'Eosinophils (EOS) Characteristics',
            'content': 'Eosinophils are granulocytes that play a key role in allergic reactions and defense against parasites. They contain distinctive red-orange granules when stained with Wright-Giemsa. Normal eosinophil count in bone marrow is 1-3%. Elevated eosinophils (eosinophilia) can indicate allergic conditions, parasitic infections, or certain hematologic malignancies like eosinophilic leukemia. Eosinophils have bilobed nuclei and prominent cytoplasmic granules.',
            'source': 'Clinical Hematology Guide, 2023'
        },
        {
            'id': 'kb_003',
            'title': 'Lymphocytes (LYT) in Bone Marrow',
            'content': 'Lymphocytes are white blood cells involved in immune responses. They include B cells, T cells, and NK cells. In bone marrow, lymphocytes typically represent 10-20% of cells. Increased lymphocytes can indicate viral infections, chronic lymphocytic leukemia (CLL), or lymphoma. Lymphocytes have round nuclei with dense chromatin and minimal cytoplasm. Atypical lymphocytes may suggest viral infections or hematologic disorders.',
            'source': 'Immunology and Hematology, 2023'
        },
        {
            'id': 'kb_004',
            'title': 'Monocytes (MON) Morphology and Function',
            'content': 'Monocytes are large white blood cells that differentiate into macrophages and dendritic cells. They have kidney-shaped or lobulated nuclei and abundant gray-blue cytoplasm. Normal monocyte count in bone marrow is 2-8%. Elevated monocytes (monocytosis) can indicate chronic infections, inflammatory diseases, or monocytic leukemia. Monocytes are phagocytic and play roles in immune defense and tissue repair.',
            'source': 'Hematology Atlas, 2023'
        },
        {
            'id': 'kb_005',
            'title': 'Neutrophils (NGS) - Most Common Granulocyte',
            'content': 'Neutrophils are the most abundant white blood cells, comprising 50-70% of circulating leukocytes. They have segmented nuclei (2-5 lobes) and fine pink-purple granules. Neutrophils are the first responders to bacterial infections and are essential for innate immunity. Increased neutrophils (neutrophilia) indicate bacterial infections, while decreased counts (neutropenia) increase infection risk. Mature neutrophils have segmented nuclei, while immature forms (bands) have horseshoe-shaped nuclei.',
            'source': 'Complete Blood Count Interpretation, 2023'
        },
        {
            'id': 'kb_006',
            'title': 'Neutrophil Immature Forms (NIF)',
            'content': 'Immature neutrophils, also called band cells or stab cells, are precursors to mature segmented neutrophils. They have horseshoe or U-shaped nuclei and are typically less than 5% of neutrophils. Increased band cells (left shift) indicate active bone marrow response to infection or inflammation. High numbers of immature forms can suggest severe infections, sepsis, or bone marrow disorders. They are important indicators of bone marrow activity.',
            'source': 'Hematology Laboratory Manual, 2023'
        },
        {
            'id': 'kb_007',
            'title': 'Promyelocytes (PMO) - Early Granulocyte Precursor',
            'content': 'Promyelocytes are early granulocyte precursors found in bone marrow. They are larger than myelocytes and have round to oval nuclei with prominent nucleoli. The cytoplasm contains primary (azurophilic) granules. Promyelocytes normally represent less than 5% of bone marrow cells. Increased promyelocytes are characteristic of acute promyelocytic leukemia (APL), a subtype of AML. APL is associated with a specific genetic translocation (t(15;17)) and requires specialized treatment.',
            'source': 'Leukemia Classification Guide, 2023'
        },
        {
            'id': 'kb_008',
            'title': 'Bone Marrow Cell Classification Clinical Significance',
            'content': 'Accurate classification of bone marrow cells is crucial for diagnosing hematologic disorders. Automated classification systems using deep learning can assist pathologists in identifying cell types with high accuracy. Each cell type has distinct morphological features that can be identified through microscopy. Classification errors can lead to misdiagnosis, so high-confidence predictions are essential. Uncertainty estimation helps identify cases requiring expert review.',
            'source': 'AI in Hematology, 2023'
        },
        {
            'id': 'kb_009',
            'title': 'High Confidence Predictions in Cell Classification',
            'content': 'High confidence predictions (>90%) in bone marrow cell classification typically indicate clear morphological features matching the predicted cell type. These predictions are more reliable and may require less expert review. However, even high-confidence predictions should be validated in clinical contexts, especially for critical diagnoses like leukemia.',
            'source': 'Clinical Decision Support Systems, 2023'
        },
        {
            'id': 'kb_010',
            'title': 'Uncertainty in Medical AI Predictions',
            'content': 'Uncertainty in AI predictions can arise from ambiguous cell morphology, overlapping features between cell types, or poor image quality. High uncertainty indicates that the model is less confident and the prediction may require expert pathologist review. Epistemic uncertainty reflects model uncertainty, while aleatoric uncertainty reflects inherent data ambiguity. Both types are important for clinical decision-making.',
            'source': 'Explainable AI in Medicine, 2023'
        }
    ]
    
    df = pd.DataFrame(knowledge_data)
    
    # Ensure directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Created knowledge base with {len(df)} entries at {save_path}")
    
    return df

