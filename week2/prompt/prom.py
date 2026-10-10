import os
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")
client = Groq(api_key=my_api_key)
model="qwen/qwen3.8-27b"

def llm_ans(prompt):
    message={
        "role":"user",
        "content":prompt
    }
    messages=[message]
    resposne = client.chat.completions.create(model=model, messages=messages)
    answer=resposne.choices[0].message.content
    return answer

bad_prompt="""
#ROLE:
you are a support assistant at a laptop/mobile company

#TASK:
You have to classify the issue in category
#constraints:
you have to classify the issue in one of the 3 categories namely billing, technical,return
#output format:
your answer should be in one word only. the one word should be one of the categories given in constraints
#EXAMPLE
for instance if a user wants a refund then the category is return
#FALLBACK
if the issue is unrelated to any of the categories, then the answer should be other
THis is a user complaint:
my marriage is broke
"""
#its not stable



print(llm_ans(bad_prompt))