import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("No api key")

client = Groq(api_key=my_api_key)

model = "qwen/qwen3.8-27b"
role = "user"
prompt1 = "Hi"
prompt2 = "explain time travel "

prompts = [prompt1, prompt2]

for prompt in prompts:
    message = {
        "role": role,
        "content": prompt
    }

    messages = [message]

    response = client.chat.completions.create(model=model, messages=messages, max_tokens=100)

    answer = response.choices[0].message.content
    
    # FIX: Extract usage directly from the response object
    usage = response.usage

    print(f"\n--- Output for '{prompt}' ---")
    print(answer)
    print(f"\n[Tokens] Prompt: {usage.prompt_tokens} | Completion: {usage.completion_tokens} | Total: {usage.total_tokens} Finish Reason: {response.choices[0].finish_reason}")