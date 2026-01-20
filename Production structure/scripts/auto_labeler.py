import sys
import os
import pandas as pd
from transformers import pipeline

# Add project root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import settings

def run_auto_labeler():
    print("Starting Auto-Labeler (Teacher Model)...")

    # SETUP PATHS
    # In a real scenario, this would be your massive 'twcs.csv' file
    # For now, we will assume you have a raw file, or we create a dummy one if missing
    raw_data_path = os.path.join(settings.BASE_DIR, 'data', 'raw', 'raw_customer_chats.csv')
    output_sentiment_path = os.path.join(settings.BASE_DIR, 'data', 'processed', 'automatically_labelled_support_data.csv')
    output_intent_path = os.path.join(settings.BASE_DIR, 'data', 'processed', 'automatically_labelled_intents.csv')

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_sentiment_path), exist_ok=True)

    # LOAD OR CREATE RAW DATA
    if os.path.exists(raw_data_path):
        print(f"   - Loading raw data from {raw_data_path}...")
        df = pd.read_csv(raw_data_path)
    else:
        print(f"Raw data not found at {raw_data_path}. Creating a small sample dataset...")
        # Create sample data so the script works immediately for testing
        df = pd.DataFrame({
            'text': [
                "I love this product!", "This is the worst service ever.", 
                "Where is my package?", "I want a refund please.", 
                "My password isn't working.", "Can you help me log in?",
                "Do you ship to Canada?", "It arrived broken."
            ] * 10 
        })
        # Save this raw file so you have it for next time
        os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
        df.to_csv(raw_data_path, index=False)

    print(f"   - Processing {len(df)} rows...")

    # INITIALIZE TEACHER MODEL (Zero-Shot)
    print("   - Loading Zero-Shot Classifier (This usually takes 30s)...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # LABEL SENTIMENT
    print("   - Labeling Sentiment...")
    candidate_labels = ["positive", "negative", "neutral"]
    
    # Run inference (This is slow on CPU)
    # We take the top label for each row
    sentiment_results = classifier(df['text'].tolist(), candidate_labels)
    df['sentiment'] = [res['labels'][0] for res in sentiment_results]
    
    # Save Sentiment CSV
    df[['text', 'sentiment']].to_csv(output_sentiment_path, index=False)
    print(f"Saved Sentiment Data to: {output_sentiment_path}")

    # LABEL INTENT
    print("   - Labeling Intents...")
    # Define the intents you want the bot to learn
    intent_labels = ["refund", "shipping", "account_issue", "technical_support", "general_inquiry"]
    
    intent_results = classifier(df['text'].tolist(), intent_labels)
    df['intent'] = [res['labels'][0] for res in intent_results]

    # Save Intent CSV
    df[['text', 'intent']].to_csv(output_intent_path, index=False)
    print(f"Saved Intent Data to: {output_intent_path}")

if __name__ == "__main__":
    run_auto_labeler()