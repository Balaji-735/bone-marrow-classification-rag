"""
RAG (Retrieval-Augmented Generation) model for generating evidence-backed explanations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rag_framework.knowledge_base.hematology_knowledge import get_knowledge_base


class SimpleRAGModel:
    """
    Simple RAG model using embedding-based retrieval and templated generation.
    """
    
    def __init__(self, knowledge_base_path=None, top_k=3):
        """
        Initialize RAG model.
        
        Args:
            knowledge_base_path: Path to knowledge base CSV
            top_k: Number of top documents to retrieve
        """
        self.top_k = top_k
        self.knowledge_base = get_knowledge_base(knowledge_base_path)
        
        # Simple keyword-based embedding (can be replaced with actual embeddings)
        self._build_keyword_index()
    
    def _build_keyword_index(self):
        """Build a simple keyword index for retrieval."""
        self.keyword_index = {}
        
        for idx, row in self.knowledge_base.iterrows():
            # Extract keywords from title and content
            text = f"{row['title']} {row['content']}".lower()
            keywords = set(text.split())
            
            for keyword in keywords:
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = []
                self.keyword_index[keyword].append(idx)
    
    def _compute_similarity(self, query: str, document: str) -> float:
        """
        Compute simple keyword-based similarity.
        
        Args:
            query: Query string
            document: Document string
            
        Returns:
            Similarity score
        """
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        if len(query_words) == 0:
            return 0.0
        
        # Jaccard similarity
        intersection = len(query_words & doc_words)
        union = len(query_words | doc_words)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve top-k relevant documents.
        
        Args:
            query: Query string
            
        Returns:
            List of retrieved documents with scores
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scores = []
        
        for idx, row in self.knowledge_base.iterrows():
            doc_text = f"{row['title']} {row['content']}"
            similarity = self._compute_similarity(query, doc_text)
            
            # Bonus for exact class name matches
            for word in query_words:
                if word in row['title'].lower() or word in row['content'].lower():
                    similarity += 0.1
            
            scores.append({
                'id': row['id'],
                'title': row['title'],
                'content': row['content'],
                'source': row['source'],
                'score': similarity
            })
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:self.top_k]
    
    def generate_explanation(self, query: str, retrieved_docs: List[Dict] = None) -> str:
        """
        Generate explanation from retrieved documents.
        
        Args:
            query: Query string
            retrieved_docs: Retrieved documents (if None, will retrieve)
            
        Returns:
            Generated explanation text
        """
        if retrieved_docs is None:
            retrieved_docs = self.retrieve(query)
        
        if len(retrieved_docs) == 0:
            return "No relevant information found in the knowledge base."
        
        # Extract key information from retrieved documents
        explanations = []
        sources = []
        
        for doc in retrieved_docs:
            if doc['score'] > 0.1:  # Only use relevant documents
                explanations.append(f"• {doc['content']}")
                sources.append(doc['source'])
        
        # Combine explanations
        explanation_text = "\n\n".join(explanations)
        
        # Add sources
        if sources:
            unique_sources = list(set(sources))
            explanation_text += f"\n\nSources: {', '.join(unique_sources)}"
        
        return explanation_text
    
    def query(self, query: str) -> Dict:
        """
        Complete RAG pipeline: retrieve and generate.
        
        Args:
            query: Query string
            
        Returns:
            Dictionary with explanation and retrieved documents
        """
        retrieved_docs = self.retrieve(query)
        explanation = self.generate_explanation(query, retrieved_docs)
        
        return {
            'query': query,
            'explanation': explanation,
            'retrieved_documents': retrieved_docs,
            'num_sources': len(retrieved_docs)
        }


def load_rag_model(knowledge_base_path=None, top_k=3):
    """
    Load and return a RAG model instance.
    
    Args:
        knowledge_base_path: Path to knowledge base
        top_k: Number of documents to retrieve
        
    Returns:
        RAG model instance
    """
    return SimpleRAGModel(knowledge_base_path, top_k)


if __name__ == "__main__":
    # Test RAG model
    rag = load_rag_model()
    
    test_query = "Explain the clinical significance of BLA cells in bone marrow"
    result = rag.query(test_query)
    
    print("Query:", result['query'])
    print("\nExplanation:")
    print(result['explanation'])
    print(f"\nRetrieved {result['num_sources']} documents")

