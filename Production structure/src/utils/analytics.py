import csv
import os
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class ExperimentLogger:
    def __init__(self):
        # Save logs in the 'logs' folder at project root
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        log_dir = os.path.join(base_path, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        self.filepath = os.path.join(log_dir, 'production_logs.csv')
        
        # Initialize file with headers if it doesn't exist
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "session_id", "variant", 
                    "user_query", "sentiment", "intent", 
                    "retrieved_context", "llm_response", 
                    "latency_seconds", "quality_score"
                ])

    def log(self, session_id, variant, query, sentiment, intent, context, response, latency, score):
        with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(), session_id, variant, 
                query, sentiment, intent, 
                context, response, 
                round(latency, 4), round(score, 4)
            ])

class QualityValidator:
    def __init__(self):
        # We reuse the RAG embedding model for validation
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def validate(self, llm_response, retrieved_context):
        if not llm_response or len(llm_response) < 5:
            return 0.0, "Too Short"
            
        # Create vectors
        embeddings = self.model.encode([llm_response, retrieved_context])
        
        # Calculate similarity
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return float(score), "Valid"