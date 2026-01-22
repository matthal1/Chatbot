from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

class ChatGenerator:
    def __init__(self):
        # We use the free local model
        model_name = "google/flan-t5-large"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        self.pipeline = pipeline(
            "text2text-generation",
            model=self.model, 
            tokenizer=self.tokenizer, 
            max_length=256,
            temperature=0.3,
            do_sample=True,
            repetition_penalty=1.2
        )

    def generate_response(self, user_query, retrieved_context, sentiment, intent):
        # Construct the prompt
        prompt = f"""
        You are a helpful Customer Support Agent. Follow these rules significantly:
        1. Answer the user's question using ONLY the Context provided below.
        2. If the Context does not contain the answer, say "I don't have that information right now."
        3. Do not make up facts.
        4. Be polite and concise.

        Context:
        {retrieved_context}
        
        User Sentiment: {sentiment}
        User Intent: {intent}
        
        User Question: {user_query}
        
        Answer:
        """
        
        response = self.pipeline(prompt)
        return response[0]['generated_text']