import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import faiss
import google.generativeai as genai

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=google_api_key)
genai_model = genai.GenerativeModel('gemini-2.0-flash')

def generate_random_documents(num_docs = 100):
    topics = ["Technology", "Science", "History", "Art", "Business"]
    documents = []

    for i in range(num_docs):
        topic = np.random.choice(topics)
        content = f"This is document {i} about {topic}. "
        content += f"It contains random information related to {topic}. "
        content += f"You can use this document to test your RAG system."

        documents.append({
            "content": content,
            "metadata": {"id": i, "topic": topic}
        })

    return documents

documents = generate_random_documents()

def get_embedding(text):
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_query"
    )
    return result['embedding']

document_texts = [doc["content"] for doc in documents]
document_embeddings = np.array([get_embedding(text) for text in document_texts])

document_embeddings = document_embeddings / np.linalg.norm(document_embeddings, axis = 1, keepdims = True)

dimension = len(document_embeddings[0])
index = faiss.IndexFlatIP(dimension)
index.add(document_embeddings)

def retrieve_documents(query, k=3):
    # Get query embedding
    query_embedding = get_embedding(query)
    query_embedding = np.array([query_embedding])

    # Normalize query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Search in the index
    scores, indices = index.search(query_embedding, k)

    # Return the top k documents
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "content": documents[idx]["content"],
            "metadata": documents[idx]["metadata"],
            "score": scores[0][i]
        })

    return results

def generate_rag_response(query):
    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(query, k=3)

    # Format context from retrieved documents
    context = "\n\n".join([doc["content"] for doc in retrieved_docs])

    # Create prompt with context
    prompt = f"""
    Context information:
    {context}

    Based on the context information, please answer the following question:
    {query}

    If the context doesn't contain relevant information, please say so.
    """

    response = genai_model.generate_content(prompt)

    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response.text
    }

# Example query
result = generate_rag_response("Tell me about technology")
print("Query:", result["query"])
print("\nResponse:", result["response"])
print("\nRetrieved Documents:")
for i, doc in enumerate(result["retrieved_documents"]):
    print(f"{i+1}. {doc['content']} (Score: {doc['score']:.4f})")

# Test with multiple questions
test_questions = [
    "What information do you have about Science?",
    "Tell me something about History",
    "What business topics are covered?"
]

for question in test_questions:
    print("\n" + "="*50)
    result = generate_rag_response(question)
    print(f"Question: {question}")
    print(f"Response: {result['response']}")