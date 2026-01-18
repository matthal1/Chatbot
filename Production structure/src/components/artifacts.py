import joblib
import os
import re

class SentimentEngine:
    def __init__(self):
        # Dynamic path finding: Go up two levels from this file to find 'artifacts'
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.artifacts_dir = os.path.join(base_path, 'artifacts')
        
        try:
            self.model = joblib.load(os.path.join(self.artifacts_dir, 'sentiment_xgboost.pkl'))
            self.vectorizer = joblib.load(os.path.join(self.artifacts_dir, 'tfidf_vectorizer.pkl'))
            self.label_encoder = joblib.load(os.path.join(self.artifacts_dir, 'label_encoder.pkl'))
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find Sentiment artifacts in {self.artifacts_dir}. Did you run the training script?")

    def predict(self, text):
        # 1. Clean
        clean_text = re.sub(r'[^a-zA-Z\s]', '', text).lower().strip()
        # 2. Vectorize
        vectorized_text = self.vectorizer.transform([clean_text])
        # 3. Predict
        pred_idx = self.model.predict(vectorized_text)[0]
        # 4. Decode
        return self.label_encoder.inverse_transform([pred_idx])[0]

class IntentEngine:
    def __init__(self):
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.artifacts_dir = os.path.join(base_path, 'artifacts')
        
        try:
            self.model = joblib.load(os.path.join(self.artifacts_dir, 'intent_logistic_regression.pkl'))
            self.vectorizer = joblib.load(os.path.join(self.artifacts_dir, 'intent_vectorizer.pkl'))
            self.label_encoder = joblib.load(os.path.join(self.artifacts_dir, 'intent_label_encoder.pkl'))
        except FileNotFoundError:
            raise FileNotFoundError("Could not find Intent artifacts. Did you run the training script?")

    def predict(self, text):
        clean_text = re.sub(r'[^a-zA-Z\s]', '', text).lower().strip()
        vectorized_text = self.vectorizer.transform([clean_text])
        pred_idx = self.model.predict(vectorized_text)[0]
        return self.label_encoder.inverse_transform([pred_idx])[0]