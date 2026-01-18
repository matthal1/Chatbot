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
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.2
        )

    def generate_response(self, user_query, retrieved_context, sentiment, intent):
        # Construct the prompt
        prompt = f"""
        Answer the following question based on the context provided. Be polite and helpful.
        
        User Sentiment: {sentiment}
        User Intent: {intent}

        Context: {retrieved_context}
        
        Question: {user_query}
        
        Answer:
        """
        
        response = self.pipeline(prompt)
        return response[0]['generated_text']