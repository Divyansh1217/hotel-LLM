# Implementation Choices & Challenges

## 📌 Approach:
- Used `FastAPI` to build a REST API.
- Embedded booking data using `FAISS` for efficient retrieval.
- Integrated an **open-source LLM** for question answering.

## 💡 Challenges:
- Finding a **suitable vector database** for embeddings.
- Handling **large dataset retrieval** efficiently.
- Fine-tuning **query processing** for better responses.

## ✅ Optimizations:
- Used `SentenceTransformers` for **faster embeddings**.
- Implemented **caching** to speed up query retrieval.

[
  {
    "query": "How much revenue did the hotel generate?",
    "expected_answer": "The total revenue is approximately $500,000."
  },
  {
    "query": "What is the cancellation rate?",
    "expected_answer": "The cancellation rate is 28.5%."
  }
]
#   h o t e l - L L M  
 #   h o t e l - L L M  
 #   h o t e l - L L M  
 