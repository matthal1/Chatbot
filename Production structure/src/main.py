import sys
import os
import time
import uuid

# This ensures Python can find your 'src' folder if you run from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your custom modules
# (If running flat in one folder, remove 'src.components.')
from src.components.guardrails import ToxicityFilter
from src.components.classifiers import SentimentEngine, IntentEngine
from src.components.rag import KnowledgeBase
from src.components.llm import ChatGenerator
from src.utils.analytics import ExperimentLogger, QualityValidator

def main():
    print("Booting up Enterprise Chatbot System...")
    
    try:
        # Load the "Ears"
        print("   - Loading Classifiers (Sentiment & Intent)...")
        sentiment_engine = SentimentEngine() 
        intent_engine = IntentEngine()

        # Load the "Eyes" (Safety)
        print("   - Loading Toxicity Filter...")
        safety_guard = ToxicityFilter()

        # Load the "Memory"
        print("   - Connecting to RAG Knowledge Base...")
        rag_system = KnowledgeBase()

        # Load the "Voice"
        print("   - Warming up LLM...")
        bot_voice = ChatGenerator()

        # Load the "Notebook" (Logging)
        print("   - Initializing Analytics...")
        logger = ExperimentLogger()
        validator = QualityValidator()

        print("System Online. Ready for queries.\n")

    except Exception as e:
        print(f"\n CRITICAL ERROR during startup: {e}")
        print("Please check that your .pkl files and ChromaDB are in the 'artifacts/' folder.")
        return

    # --- 3. THE CONVERSATION LOOP ---
    session_id = str(uuid.uuid4())
    print(f"--- Session ID: {session_id} ---")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        # A. Get User Input
        user_input = input("User: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Bot: Goodbye! Have a great day.")
            break
        
        if not user_input:
            continue

        start_time = time.time()

        # B. Step 0: Safety Guardrail (Fail Fast)
        is_safe, reason = safety_guard.check_safety(user_input)
        if not is_safe:
            print(f"Bot: I cannot respond to that. ({reason})")
            # Log the rejected message for auditing
            logger.log(session_id, "safety_block", user_input, "N/A", "N/A", "N/A", reason, 0, 0)
            continue

        # C. Step 1: Classification (The "Ears")
        # Run these in parallel in a real app, but sequential is fine here
        sentiment = sentiment_engine.predict(user_input)
        intent = intent_engine.predict(user_input)
        
        print(f"     [Debug] Sentiment: {sentiment} | Intent: {intent}")

        # D. Step 2: Retrieval (The "Memory")
        # We search specifically for policies related to the detected intent
        retrieved_context = rag_system.search(user_input)
        
        # E. Step 3: Generation (The "Voice")
        # We pass sentiment so the bot knows if it should be apologetic or happy
        raw_response = bot_voice.generate_response(
            user_query=user_input,
            retrieved_context=retrieved_context,
            sentiment=sentiment,
            intent=intent
        )

        # F. Step 4: Quality Validation (The "Editor")
        # Check if the LLM hallucinated
        quality_score, validity_reason = validator.validate(raw_response, retrieved_context)
        
        # G. Final Output Logic
        final_output = raw_response
        
        # If quality is too low, override with a fallback message
        if quality_score < 0.4:
            print(f"     [Debug] Low Quality Detected ({quality_score:.2f}). Fallback triggered.")
            final_output = "I'm not 100% sure about that based on our current policies. Let me connect you with a human agent to be safe."

        # H. Log Everything (The "MLOps")
        latency = time.time() - start_time
        logger.log(
            session_id=session_id,
            variant="v1_production",
            query=user_input,
            sentiment=sentiment,
            intent=intent,
            context=retrieved_context,
            response=final_output,
            latency=latency,
            score=quality_score
        )

        print(f"Bot: {final_output}")
        print("-" * 50)

if __name__ == "__main__":
    main()