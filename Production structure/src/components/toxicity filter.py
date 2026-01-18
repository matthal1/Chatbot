import os
from transformers import pipeline

class ToxicityFilter:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        # We load the model once when the class is initialized
        # 'unitary/toxic-bert' is the industry standard for this
        self.classifier = pipeline(
            "text-classification", 
            model="unitary/toxic-bert", 
            top_k=None 
        )

    def check_safety(self, text):
        """
        Returns (True, "Safe") or (False, "Reason")
        """
        results = self.classifier(text)
        scores = results[0]
        
        for category in scores:
            if category['score'] > self.threshold:
                reason = f"Blocked due to {category['label']} ({round(category['score'], 2)})"
                return False, reason
        
        return True, "Safe"