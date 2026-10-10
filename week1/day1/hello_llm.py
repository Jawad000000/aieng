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
prompt1 ="hi"
prompt2 ="explain time travel"
prompt3="why liverpoolfc is so successfull in 1000 words essay"

prompts=[prompt1, prompt2, prompt3]

for prompt in prompts:
    message = {
        "role":role,
        "content":prompt
    }

    messages = [message]

    response = client.chat.completions.create(model=model, messages=messages)

    answer = response.choices[0].message.content

print(answer)