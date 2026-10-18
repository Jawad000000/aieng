import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client=QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)
print("connected to client")

COLLECTION_NAME="knowledge"
EMBEDDING_SIZE=384

if client.collection_exists(COLLECTION_NAME):
    print(f"deleting the collection : {COLLECTION_NAME}")

    client.delete_collection(COLLECTION_NAME)

#create colelction

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE,
    ),
)

print(f"Created collections: {COLLECTION_NAME}")
print(f"vector size: {EMBEDDING_SIZE}")
print(f"Distance:COSINE")

#load the knowledge 

with open("knowledege.txt","r", encoding="utf-8") as f:
    documents=[
        line.strip()
        for line in f
        if line.strip()
    ]

print(f"Loaded{len(documents)} documents")

#embeddings
print("doing embedding")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings=model.encode(documents)

print(f"generated{len(embeddings)} embeddings")

#create qdrant points

points=[]
for i, embeddings in enumerate(embeddings):
    point = PointStruct(
        id= i+1,
        vector= embeddings.tolist(),

        payload={
            "text":documents[i]
        }
    )
    points.append(point)

#upload to qdrant

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print("uploaded")

#search qdrant

def search(query, top_k=3):

    query_vector = model.encode(query).tolist()

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    return result

query = "How many vacation days do I get?"

results = search(query, top_k=3)

print("\nSearch results:")

for result in results:
    print(f"Score: {result.score:.3f}")
    print(result.payload["text"])
    print()


# ============================================================
# PART 10 — CONNECT TO GROQ
# ============================================================

groq_client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# PART 11 — ASK THE LLM
# ============================================================

def ask_llm(question, context):

    prompt = f"""
Answer the question using only the information provided below.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I don't know based on the provided information."
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# ============================================================
# PART 12 — COMPLETE RAG PIPELINE
# ============================================================

question = "How many vacation days do I get?"

results = search(question, top_k=3)


# Extract text from the search results
context = "\n".join(
    result.payload["text"]
    for result in results
)


answer = ask_llm(question, context)


print("\nFinal Answer:")
print(answer)