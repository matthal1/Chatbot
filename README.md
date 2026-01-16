This project objective is to emulate a enterprise chatbot by building a multilevel system. 
The goal of this system is to create a cost effective solution to using a basic chatgpt wrapper. Rather than relying on a useful but expenisve solution, this idea would enable a company to use their own in house solution to reduce costs.

Why does this matter?
  We want to create a system that include gaurdrails and has low latency when used in a industrial environment. This will allow for better users interaction and better customer support.
What is the structure of the project?
  Toxicity filter (BERT) -> Sentiment Classifer (XGBoost) -> Intent Classification (Logistic Regression) -> Context Retrival(Vector dB/RAG) -> Response Generation(GPT-4)
