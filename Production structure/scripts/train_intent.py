import sys
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import settings

def train_intent_model():
    print("Starting Intent Training...")

    # Construct the path to your processed data file
    # Make sure this filename matches whatever your Auto-Labeler outputted for intents
    data_path = os.path.join(settings.BASE_DIR, 'data', 'processed', 'automatically_labelled_intents.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Critical Error: Could not find training data at {data_path}")
        
    print(f"   - Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Clean up column names to ensure they match expectations
    # (Adjust 'intent_label' or 'clean_text' if your CSV headers are different)
    if 'intent_label' in df.columns:
        df.rename(columns={'intent_label': 'intent'}, inplace=True)
    if 'clean_text' in df.columns:
        df.rename(columns={'clean_text': 'text'}, inplace=True)

    # Sanity Check: Drop rows where text or intent is missing
    initial_count = len(df)
    df.dropna(subset=['text', 'intent'], inplace=True)
    if len(df) < initial_count:
        print(f"   - Dropped {initial_count - len(df)} rows with missing data.")

    print(f"   - Training on {len(df)} examples.")

    # VECTORIZATION
    print("   - Vectorizing...")
    # We use .astype(str) to prevent crashes on non-string inputs
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df['text'].astype(str))

    # ENCODE LABELS
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['intent'])
    
    # Print the classes found so you can verify them
    print(f"   - Detected Intents: {list(encoder.classes_)}")

    # TRAIN MODEL
    print("   - Training Logistic Regression...")
    # Increased max_iter to 1000 to ensure convergence on larger real datasets
    model = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000)
    model.fit(X, y)

    # SAVE ARTIFACTS
    print(f"   - Saving artifacts to {settings.ARTIFACTS_DIR}...")
    os.makedirs(settings.ARTIFACTS_DIR, exist_ok=True)

    joblib.dump(model, settings.INTENT_MODEL_PATH)
    joblib.dump(vectorizer, settings.INTENT_VECTORIZER_PATH)
    joblib.dump(encoder, settings.INTENT_LABEL_PATH)

    print("Intent Model Built Successfully!")

if __name__ == "__main__":
    train_intent_model()