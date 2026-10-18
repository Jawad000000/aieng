import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=my_api_key)
groq_model = "openai/gpt-oss-20b"
model = SentenceTransformer("all-MiniLM-L6-v2")#384

documents = [
    "Employees receive 24 days of paid leave per year.",
    "Employees work from the office on Tuesday, Wednesday and Thursday. Monday and Friday are optional work-from-home days.",
    "Employees receive Rs 3000 per month for gym reimbursement.",
    "Employees can claim Rs 2000 per month for home internet.",
    "Employees have a 90 day notice period."
]

document_embedding = model.encode(documents)
def cosine_similarity(a,b):
    return np.dot(a,b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


#QUEry also has a array
#i can score this with all the doc array and witht the highest score we will as context

def retrieve(query_embedding):

    scores = []

    for i, document in enumerate(document_embedding):

        score = cosine_similarity(query_embedding, document)
        scores.append((score, documents[i]))

        scores.sort(reverse=True)

        return scores[0]

def ask_llm(question, context):
    
    sys_prompt=f"""answer in one line only , and answer based only on this context. do not hallucinate. Context: {context}"""

    system_message={
        "role":"system",
        "content":sys_prompt
    }
    message ={
        "role": "user",
        "content":question
    }

    messages=[system_message,message]

    response=client.chat.completions.create(model=groq_model, messages=messages)
    answer=response.choices[0].message.content
    return answer

query = "How many vacation days do employees get?"

query_embedding = model.encode(query)

score, context = retrieve(query_embedding)

answer = ask_llm(query,context)
print(answer)