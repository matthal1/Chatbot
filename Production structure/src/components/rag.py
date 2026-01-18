import chromadb
from chromadb.utils import embedding_functions
import os

class KnowledgeBase:
    def __init__(self):
        # Path to the persistent database
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        db_path = os.path.join(base_path, 'artifacts', 'chroma_db_data')
        
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Using the same embedding function used during ingestion
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get the collection. We assume it was created by your build_rag_db.py script.
        try:
            self.collection = self.client.get_collection(
                name="company_knowledge_base",
                embedding_function=self.ef
            )
        except Exception:
            # Fallback if collection doesn't exist yet
            print("Warning: Knowledge Base not found. Creating empty one.")
            self.collection = self.client.create_collection(
                name="company_knowledge_base",
                embedding_function=self.ef
            )

    def search(self, query, n_results=1):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if results['documents'] and results['documents'][0]:
            return results['documents'][0][0]
        else:
            return "No specific policy found for this issue."