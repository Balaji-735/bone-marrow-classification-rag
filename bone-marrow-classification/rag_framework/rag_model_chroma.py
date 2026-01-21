"""
Enhanced RAG model using Chroma vector database and embedding-based retrieval.
Integrates with rag-tutorial-v2-main PDF dataset.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain_community.embeddings.ollama import OllamaEmbeddings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Path to rag-tutorial-v2-main directory
RAG_TUTORIAL_PATH = project_root.parent / "rag-tutorial-v2-main"
CHROMA_PATH = RAG_TUTORIAL_PATH / "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


class ChromaRAGModel:
    """
    Enhanced RAG model using Chroma vector database for retrieval.
    Uses PDF documents from rag-tutorial-v2-main as knowledge base.
    """
    
    def __init__(self, chroma_path=None, top_k=3, use_llm=False, llm_model="mistral"):
        """
        Initialize Chroma-based RAG model.
        
        Args:
            chroma_path: Path to Chroma database (defaults to rag-tutorial-v2-main/chroma)
            top_k: Number of top documents to retrieve
            use_llm: Whether to use LLM for generation (default: False, uses templated generation)
            llm_model: LLM model name if use_llm=True
        """
        self.top_k = top_k
        self.use_llm = use_llm
        
        # Set Chroma path
        if chroma_path is None:
            chroma_path = CHROMA_PATH
        
        self.chroma_path = Path(chroma_path)
        
        # Initialize embedding function
        self.embedding_function = OllamaEmbeddings(model="nomic-embed-text")
        
        # Initialize Chroma database
        try:
            self.db = Chroma(
                persist_directory=str(self.chroma_path),
                embedding_function=self.embedding_function
            )
            # Test if database is accessible
            try:
                _ = self.db.get()
            except Exception as db_error:
                # If database is corrupted, provide helpful error message
                error_msg = (
                    f"Chroma database at {self.chroma_path} appears to be corrupted.\n"
                    f"To fix this:\n"
                    f"1. Delete the chroma folder: {self.chroma_path}\n"
                    f"2. Run: cd rag-tutorial-v2-main && python populate_database.py\n"
                    f"Error details: {db_error}"
                )
                raise ValueError(error_msg)
        except ValueError:
            raise
        except Exception as e:
            error_msg = (
                f"Failed to load Chroma database at {self.chroma_path}.\n"
                f"Please ensure the database exists. Run 'python populate_database.py' in rag-tutorial-v2-main first.\n"
                f"Error: {e}"
            )
            raise ValueError(error_msg)
        
        # Initialize LLM if needed
        if self.use_llm:
            self.llm = Ollama(model=llm_model)
        else:
            self.llm = None
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve top-k relevant documents using embedding-based similarity with keyword boosting.
        
        Args:
            query: Query string
            
        Returns:
            List of retrieved documents with scores
        """
        # Get more results for better filtering (2x for keyword boosting)
        results = self.db.similarity_search_with_score(query, k=self.top_k * 2)
        
        # Extract query terms for keyword matching
        query_terms = set(query.lower().split())
        
        # Format results with keyword boosting
        retrieved_docs = []
        for doc, score in results:
            # Convert distance score to similarity score (lower distance = higher similarity)
            similarity_score = 1.0 / (1.0 + score) if score > 0 else 1.0
            
            # Keyword boost: check how many query terms appear in content
            content_lower = doc.page_content.lower()
            term_matches = sum(1 for term in query_terms if len(term) > 2 and term in content_lower)
            
            # Boost score for documents containing query terms
            keyword_boost = min(term_matches * 0.15, 0.5)  # Max boost of 0.5
            boosted_score = similarity_score + keyword_boost
            
            retrieved_docs.append({
                'id': doc.metadata.get('id', 'Unknown'),
                'title': doc.metadata.get('source', 'Unknown Source'),
                'content': doc.page_content,
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', None),
                'score': boosted_score,
                'distance': score,
                'keyword_matches': term_matches
            })
        
        # Sort by boosted score and return top-k
        retrieved_docs.sort(key=lambda x: x['score'], reverse=True)
        return retrieved_docs[:self.top_k]
    
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
        
        if self.use_llm and self.llm:
            # Use LLM for generation
            context_text = "\n\n---\n\n".join([doc['content'] for doc in retrieved_docs])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query)
            explanation = self.llm.invoke(prompt)
        else:
            # Use templated generation with better filtering
            explanations = []
            sources = []
            
            # Filter by relevance and content quality
            relevant_docs = []
            for doc in retrieved_docs:
                # Check if content mentions cell type or clinical terms
                content_lower = doc['content'].lower()
                query_terms = query.lower().split()
                
                # Count how many query terms appear in content
                term_matches = doc.get('keyword_matches', 0)
                
                # Only include if has good match or reasonable score
                if doc['score'] > 0.001 and (term_matches >= 2 or doc['score'] > 0.01):
                    relevant_docs.append(doc)
            
            if not relevant_docs:
                return "No highly relevant information found. The retrieved documents may focus on methodology rather than clinical descriptions. Consider adding clinical textbooks to the knowledge base."
            
            for doc in relevant_docs:
                source_info = doc['source']
                if doc['page'] is not None:
                    source_info += f" (Page {doc['page']})"
                
                explanations.append(f"• {doc['content']}")
                sources.append(source_info)
            
            # Combine explanations
            explanation = "\n\n".join(explanations)
            
            # Add sources
            if sources:
                unique_sources = list(set(sources))
                explanation += f"\n\nSources: {', '.join(unique_sources)}"
        
        return explanation
    
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


def load_rag_model(knowledge_base_path=None, top_k=3, use_chroma=True, 
                   chroma_path=None, use_llm=False, llm_model="mistral"):
    """
    Load and return a RAG model instance.
    
    Args:
        knowledge_base_path: Path to knowledge base (ignored if use_chroma=True)
        top_k: Number of documents to retrieve
        use_chroma: Whether to use Chroma vector DB (default: True)
        chroma_path: Path to Chroma database
        use_llm: Whether to use LLM for generation
        llm_model: LLM model name
        
    Returns:
        RAG model instance
    """
    if use_chroma:
        return ChromaRAGModel(chroma_path=chroma_path, top_k=top_k, 
                              use_llm=use_llm, llm_model=llm_model)
    else:
        # Fallback to original SimpleRAGModel
        from rag_framework.rag_model import SimpleRAGModel
        return SimpleRAGModel(knowledge_base_path, top_k)


if __name__ == "__main__":
    # Test Chroma RAG model
    print("Testing Chroma RAG model...")
    rag = load_rag_model(use_chroma=True, top_k=3, use_llm=False)
    
    test_query = "Explain the clinical significance of BLA cells in bone marrow"
    result = rag.query(test_query)
    
    print("Query:", result['query'])
    print("\nExplanation:")
    print(result['explanation'])
    print(f"\nRetrieved {result['num_sources']} documents")
    print("\nRetrieved Documents:")
    for i, doc in enumerate(result['retrieved_documents'], 1):
        print(f"\n[{i}] Score: {doc['score']:.4f}")
        print(f"Source: {doc['source']}")
        print(f"Content: {doc['content'][:200]}...")

