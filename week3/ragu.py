import os
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError

client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-120b"


#step1
knowledge_base={
    "know": "padho with pratyush is a youtuber, his name is pratyush",
    "age": "age of pratyus is 25",
    "work":"he make youtube videos for dsa and cs prep for ai/sde interview and guide"

}
#step 2 retriveal

def retrieve_info(question):
    question=question.lower()

    if "know" in question:
        return knowledge_base["know"]

    elif "age" in question:
        return knowledge_base["age"]
    else:
        return knowledge_base['work']

def ask_llm(question):
    context = retrieve_info(question)
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

    response=client.chat.completions.create(model=model, messages=messages)
    answer=response.choices[0].message.content
    return answer

question="what is age"
print(ask_llm(question))