import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

load_dotenv()
my_api_key= os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("No api key")

client = Groq(api_key=my_api_key)
 
model = "qwen/qwen3.8-27b"
role ="user"

#structure
class Ticket(BaseModel):
    name:str
    email:str
    issue:str

schema = Ticket.model_json_schema()
response_format = {
    "type": "json_object"
}

system_prompt =f"""
Extract the personal information from the ticket strictly based on this schema and give a json output.
{schema}
"""
message_system={
    "role": "system",
    "content":system_prompt
}

text="hello, my name is jawad, i have an iphone which is not working at all,i have 2 legs and 6 mouths, my house is in delhi, my email is abc@gmail.com. my contact numeber is 7451230"

prompt = f"""
This is a customer ticket. please extract the personal information from this.
{text}
"""


message = {
"role":role,
"content":prompt
}

messages = [message_system, message]

response = client.chat.completions.create(model=model, messages=messages, response_format=response_format)

answer = response.choices[0].message.content

print(answer)

#how to extract

import json
raw_json = answer
data_file = json.loads(raw_json)
ticket = Ticket(**data_file)
print(ticket.name)
print(ticket.email)
print(ticket.issue)

#