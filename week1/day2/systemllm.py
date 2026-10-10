import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key= os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("No api key")

client = Groq(api_key=my_api_key)

model = "qwen/qwen3.8-27b"
role ="user"
prompt = "suggest a name for my food company"

message_system={
    "role":"system",
    "content":"you are a brand manager, suggest name for my food brand, name should be in one word, suggest only one name"
}

message = {
    "role":role,
    "content":prompt
}

messages = [message_system, message]

response = client.chat.completions.create(model=model, messages=messages, temperature =2)

answer = response.choices[0].message.content

print(answer)