import sys
import os
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

# Adds the project root to python path so we can import 'config'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import settings

def train_sentiment_model():
    print("Starting Sentiment Training...")

    # LOAD DATA
    # This points to the file created by your Auto-Labeling step
    data_path = os.path.join(settings.BASE_DIR, 'data', 'processed', 'automatically_labelled_support_data.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find data at {data_path}. Did you run the Auto-Labeler?")

    print(f"   - Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Standardize column names
    if 'sentiment_label' in df.columns:
        df.rename(columns={'sentiment_label': 'sentiment'}, inplace=True)
        
    if 'clean_text' in df.columns:
        df.rename(columns={'clean_text': 'text'}, inplace=True)
    
    print(f"   - Loaded {len(df)} training examples")

    # VECTORIZATION
    print("   - Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    # Use the 'text' column we just renamed
    X = vectorizer.fit_transform(df['text'].astype(str)) # Added .astype(str) to prevent errors if text is empty

    # ENCODE LABELS
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['sentiment'])

    # TRAIN/TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # CALCULATE WEIGHTS (Handle Class Imbalance)
    print("   - Calculating Class Weights...")
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # TRAIN XGBOOST
    print("   - Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # SAVE ARTIFACTS
    print(f"   - Saving artifacts to {settings.ARTIFACTS_DIR}...")
    
    # Ensure directory exists
    os.makedirs(settings.ARTIFACTS_DIR, exist_ok=True)
    
    joblib.dump(model, settings.SENTIMENT_MODEL_PATH)
    joblib.dump(vectorizer, settings.SENTIMENT_VECTORIZER_PATH)
    joblib.dump(encoder, settings.SENTIMENT_LABEL_PATH)
    
    print("Sentiment Model Built Successfully!")

if __name__ == "__main__":
    train_sentiment_model()