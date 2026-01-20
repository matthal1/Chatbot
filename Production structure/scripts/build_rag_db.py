import sys
import os
import chromadb
from chromadb.utils import embedding_functions

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import settings

def build_knowledge_base():
    print("🚀 Building RAG Knowledge Base...")

    documents = []
    
    # --- 1. READ & CHUNK DATA ---
    policy_dir = os.path.join(settings.BASE_DIR, 'data', 'raw', 'policies')
    
    # In production, fail if data is missing. Don't fallback to mock data silently.
    if not os.path.exists(policy_dir):
        raise FileNotFoundError(f"CRITICAL: No policy folder found at {policy_dir}. Cannot build database.")

    # Loop through every .txt file
    files_found = [f for f in os.listdir(policy_dir) if f.endswith(".txt")]
    if not files_found:
        raise ValueError(f"CRITICAL: The folder {policy_dir} exists but is empty.")

    print(f"   - Found {len(files_found)} policy files.")

    for filename in files_found:
        file_path = os.path.join(policy_dir, filename)
        
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()
        
        # --- CHUNKING LOGIC (Production Requirement) ---
        # We split by double newlines to treat each paragraph as a separate searchable chunk.
        # This prevents the 256-token limit from cutting off your data.
        chunks = full_text.split('\n\n')
        
        # Base ID from filename (e.g., "refund_policy")
        base_id = filename.split('.')[0]
        
        # Guess category
        category = "refund" if "refund" in filename else "shipping" if "shipping" in filename else "general"

        for i, chunk in enumerate(chunks):
            # Skip empty whitespace chunks
            if chunk.strip(): 
                documents.append({
                    "id": f"{base_id}_chunk_{i}", # Unique ID for every paragraph
                    "text": chunk.strip(),
                    "category": category
                })

    print(f"   - processed into {len(documents)} searchable chunks.")

    # --- 2. SETUP CHROMADB ---
    print(f"   - Connecting to database at {settings.CHROMA_DB_PATH}...")
    
    client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # --- 3. RESET COLLECTION ---
    try:
        client.delete_collection(name=settings.CHROMA_COLLECTION_NAME)
        print("   - Deleted old collection to start fresh.")
    except ValueError:
        pass 

    collection = client.create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=ef
    )

    # --- 4. INDEXING ---
    if documents:
        print("   - Indexing documents...")
        # Add in batches (Chroma handles large lists better in batches, but this is fine for <10k docs)
        collection.add(
            ids=[doc['id'] for doc in documents],
            documents=[doc['text'] for doc in documents],
            metadatas=[{"category": doc['category']} for doc in documents]
        )
        print(f"Knowledge Base Built! Indexed {len(documents)} chunks.")
    else:
        print("Warning: No valid text chunks were found to index.")

if __name__ == "__main__":
    build_knowledge_base()