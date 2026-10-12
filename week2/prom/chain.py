import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key= os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("where is api key")

client=Groq(api_key=my_api_key)

model="qwen/qwen3-32b"

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