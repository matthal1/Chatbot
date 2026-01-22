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
        # Create a more balanced and diverse dataset
        data = {
            'text': [
                # --- POSITIVE (20 examples) ---
                "I absolutely love this product, it works great!",
                "The shipping was incredibly fast, thank you!",
                "Customer service was so helpful and kind.",
                "Best purchase I've made all year.",
                "The quality of the material is fantastic.",
                "I am very happy with my order.",
                "You guys are the best, thanks for the help!",
                "The app is so easy to use now, great update.",
                "My refund was processed immediately, very impressed.",
                "I really appreciate the quick response.",
                "The packaging was beautiful and secure.",
                "Everything arrived perfect, thanks!",
                "I'm a huge fan of your new collection.",
                "Your team went above and beyond for me.",
                "Excellent experience, will buy again.",
                "The instructions were clear and easy to follow.",
                "Finally a support team that actually listens.",
                "Wow, that was lighter/faster than I expected!",
                "Great job on resolving my issue so quickly.",
                "I love the new features you added.",

                # --- NEGATIVE (20 examples) ---
                "This is the worst service I have ever received.",
                "My package arrived completely crushed and broken.",
                "I've been waiting for a refund for weeks.",
                "Nobody is answering my emails, this is frustrating.",
                "The product broke after one day of use.",
                "I want to speak to a manager immediately.",
                "Your website is broken and I can't log in.",
                "Shipping is taking forever, where is my stuff?",
                "I am very disappointed with the quality.",
                "Don't buy from here, it's a scam.",
                "The size guide is completely wrong.",
                "I was charged twice for the same item!",
                "Rude customer service agent hung up on me.",
                "The app keeps crashing on my phone.",
                "I demand a full refund right now.",
                "It's been a month and I still don't have my order.",
                "This doesn't look anything like the picture.",
                "Stop sending me spam emails.",
                "I can't believe how bad this experience was.",
                "Your return policy is unfair and confusing.",

                # --- NEUTRAL (20 examples) ---
                "How long does shipping normally take?",
                "What is your return policy?",
                "Do you ship to Canada?",
                "I need to reset my password.",
                "Where can I find the tracking number?",
                "Is this item in stock?",
                "Can I change my shipping address?",
                "How do I cancel my subscription?",
                "What forms of payment do you accept?",
                "Are you open on weekends?",
                "Does this come with a warranty?",
                "I have a question about my account.",
                "How do I clear my cart?",
                "Is there a physical store location?",
                "Do you offer gift cards?",
                "How do I contact support?",
                "What is the difference between these two models?",
                "I didn't receive a confirmation email.",
                "Can I track my order without logging in?",
                "When will the sale end?"
            ] * 5  # Replicate 5 times to get ~300 rows
        }
        df = pd.DataFrame(data)
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