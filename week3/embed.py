import os
from dotenv import load_dotenv
import numpy as np 
from groq import Groq
from pathlib import Path
from sentence_transformers import SentenceTransformer

#sentence transformer has its own model
def cosine_similarity(a,b):
    return np.dot(a,b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


model = SentenceTransformer("all-MiniLM-L6-v2")#384
text="Machine learning is fun"

# embedding=model.encode(text)
# print(embedding.shape)
# print(embedding[:10])

t1="there are 24 paid leaves"
t2="there are 24 vacation days"

e1=model.encode(t1)
e2=model.encode(t2)
print(cosine_similarity(e1,e2))