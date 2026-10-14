import os
from pathlib import Path
from time import sleep
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key= os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("where is api key")

client=Groq(api_key=my_api_key)

model = "openai/gpt-oss-120b"

JD="""
Job Title: Senior Data Engineer
Company: Apex Analytics
Location: Remote

About the Role:
Apex Analytics is looking for a Senior Data Engineer to build and maintain scalable ETL/ELT pipelines processing multi-terabyte data streams for enterprise reporting.

Key Responsibilities:
- Design, construct, and maintain batch and real-time data pipelines using Python, PySpark, and Apache Airflow.
- Build and optimize data warehouse models in Snowflake and PostgreSQL.
- Implement automated data quality checks and validation frameworks using Great Expectations or Pydantic.
- Containerize pipeline services with Docker and manage deployments on AWS (S3, ECS, Redshift).

Requirements:
- 5+ years of professional experience in data engineering or software development.
- Expert proficiency in Python, SQL, and distributed data processing frameworks (PySpark / Delta Lake).
- Hands-on experience orchestrating workflows with Apache Airflow or Prefect.
- Solid understanding of data modeling techniques (Star Schema, Data Vault).
"""

RESUME="""
ALEX MORGAN
Email: alex.morgan@email.com | Location: Austin, TX

SUMMARY
Data Engineer with 3.5 years of experience building scalable data pipelines, data warehouses, and analytics infrastructure. Proficient in Python, SQL, Apache Airflow, and Cloud Data Warehouses.

TECHNICAL SKILLS
- Languages: Python, SQL, Bash
- Data Engineering: Apache Airflow, PySpark, Pandas, ETL/ELT Pipelines
- Databases & Warehouses: PostgreSQL, Snowflake, AWS Redshift
- Tools & Cloud: Docker, Git, AWS (S3, EC2), dbt

PROFESSIONAL EXPERIENCE
Data Engineer | DataPulse Systems (2023 – Present)
- Designed and automated 15+ daily ETL pipelines using Apache Airflow and Python, ingesting over 50GB of daily event data.
- Built a staging and analytics data warehouse on Snowflake, optimizing SQL queries to reduce execution time by 30%.
- Containerized workflow execution tasks using Docker to achieve consistency across dev and production environments.

Junior Data Analyst | TechCorp (2021 – 2023)
- Maintained SQL queries and PostgreSQL database views for internal operational dashboards.
- Automated weekly CSV processing scripts using Python and Pandas.

EDUCATION
B.S. in Computer Science | University of Texas (2021)
"""
def ask_llm(system_prompt, user_prompt):
    sys_messsage={
        "role":"system",
        "content": system_prompt
    }

    user_message={
        "role":"user",
        "content": user_prompt
    }

    messages = [sys_messsage, user_message]
    response = client.chat.completions.create(model=model, messages=messages)
    answer = response.choices[0].message.content
    return answer

def resume_extract():
    #extract skills from resume
    system_prompt="""
    YOU are a professional hr assistant. extract the skills from the candidates resume provided.
    only return the skills no other things. do not invent things by yourself
    give the answers sperated by commas only return true information and only reuturn skills
    """

    user_prompt=f"""
    extract the skills from this resume
    {RESUME}
    """
    return ask_llm(system_prompt, user_prompt)

def jd_extract():
    #extract skills from resume
    system_prompt=f"""
    YOU are a professional hr assistant. extract the skills from the job description provided.
    only return the skills no other things. do not invent things by yourself
    """

    user_prompt=f"""
    extract the skills from this JD
    {JD}
    """
    return ask_llm(system_prompt, user_prompt)


def match_skills(candidate, jd):
    system_prompt="""
    you are a professional hr assistant. compare skills of candidate and skills required in the jd and produce
    final score between 1 and 100 and produce whether the candidate is a good fit for the role
    """

    user_prompt=f"""
    TASK:compare and match the skills
    JD:
    {jd}
    candidate:
    {candidate}
    output format: only mention crisp points dont give explanations
    """

    return ask_llm(system_prompt, user_prompt)

candidate = resume_extract()
print(candidate)
sleep(2)
jd = jd_extract()
print(jd)
score = match_skills(candidate, jd)
print(score)
