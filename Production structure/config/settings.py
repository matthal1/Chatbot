import os

# This gets the absolute path of your project root (Production structure)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define where your artifacts and logs live
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Classifiers
SENTIMENT_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "sentiment_xgboost.pkl")
SENTIMENT_VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl")
SENTIMENT_LABEL_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")

INTENT_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "intent_logistic_regression.pkl")
INTENT_VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "intent_vectorizer.pkl")
INTENT_LABEL_PATH = os.path.join(ARTIFACTS_DIR, "intent_label_encoder.pkl")

# Vector Database
CHROMA_DB_PATH = os.path.join(ARTIFACTS_DIR, "chroma_db_data")
CHROMA_COLLECTION_NAME = "company_knowledge_base"

# LLM Settings
LLM_MODEL_NAME = "google/flan-t5-large"
LLM_MAX_LENGTH = 256
LLM_TEMPERATURE = 0.7
LLM_REPETITION_PENALTY = 1.2

# Guardrail Settings
TOXICITY_THRESHOLD = 0.7
TOXICITY_MODEL_NAME = "unitary/toxic-bert"

# Validator Settings
QUALITY_THRESHOLD = 0.4 # Below this, we trigger the "I don't know" fallback

# --- 4. SECRETS (Optional) ---
# In a real app, load these from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")