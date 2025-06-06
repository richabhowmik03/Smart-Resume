import os
import getpass
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import Chroma  # Updated import
from langchain_core.output_parsers import JsonOutputParser
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Set up environment
def load_env():
    load_dotenv()

def set_groq_api_key():
    if "GROQ_API_KEY" not in os.environ:
        os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

def get_llm(model="llama-3.1-8b-instant", temperature=0, max_tokens=None, timeout=None, max_retries=2):
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )

def load_job_posting():
    job_input_type = input("Type 'link' to provide a job posting URL, or 'text' to paste the job description: ").strip().lower()
    if job_input_type == "link":
        job_url = input("Paste the job posting URL: ").strip()
        loader = WebBaseLoader(job_url)
        docs = loader.load()
        job_posting_text = docs[0].page_content
    elif job_input_type == "text":
        print("Paste the full job description below. End input with an empty line:")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        job_posting_text = "\n".join(lines)
    else:
        raise ValueError("Invalid input. Please type 'link' or 'text'.")
    return job_posting_text

def extract_job_postings(llm, job_posting_text):
    prompt_extract = PromptTemplate.from_template(
        """
You are a helpful assistant.
Your job is to extract job postings from the following scraped text. 
Return ONLY a JSON list of job postings. Each posting must have:

- `role`: the job title
- `experience`: required experience (if mentioned)
- `skills`: a list of key skills mentioned
- `description`: a summary of the role

IMPORTANT:
- Output ONLY valid JSON.
- DO NOT include any code, markdown, or explanation.
- DO NOT wrap the JSON in ``` or any other format.

### SCRAPED TEXT:
{page_data}
        """
    )
    chain_extract = prompt_extract | llm 
    res = chain_extract.invoke(input={'page_data': job_posting_text})
    json_parser = JsonOutputParser()
    json_res = json_parser.parse(res.content)
    return json_res

def load_resume(pdf_path):
    resume_loader = PyPDFLoader(pdf_path)
    resume_docs = resume_loader.load()
    return resume_docs

def split_texts(resume_text, job_description_str, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    resume_chunks = text_splitter.split_documents([Document(page_content=resume_text)])
    job_chunks = text_splitter.split_documents([Document(page_content=job_description_str)])
    return resume_chunks, job_chunks

def get_embedding_model(model_name="all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

def create_vectorstores(resume_chunks, job_chunks, embedding_model):
    resume_db = Chroma.from_documents(resume_chunks, embedding_model, collection_name="resume_chunks")
    job_db = Chroma.from_documents(job_chunks, embedding_model, collection_name="job_chunks")
    return resume_db, job_db

def find_similar_chunks(resume_db, job_chunks, k=2):
    results = []
    for job_chunk in job_chunks:
        similar_resume_chunks = resume_db.similarity_search_with_score(job_chunk.page_content, k=k)
        results.append({
            "job_chunk": job_chunk,
            "similar_resume_chunks": similar_resume_chunks
        })
    return results

def compute_cosine_similarity(resume_chunks, job_chunks, embedding_model):
    resume_texts = [doc.page_content for doc in resume_chunks]
    job_texts = [doc.page_content for doc in job_chunks]
    resume_embeddings = embedding_model.embed_documents(resume_texts)
    job_embeddings = embedding_model.embed_documents(job_texts)
    similarity_matrix = cosine_similarity(resume_embeddings, job_embeddings)
    return similarity_matrix, resume_embeddings, job_embeddings

def get_most_relevant_resume_chunks(job_chunks, resume_chunks, job_embeddings, resume_embeddings):
    most_relevant_resume_chunks = []
    for i, job_vector in enumerate(job_embeddings):
        similarities = cosine_similarity([job_vector], resume_embeddings)[0]
        best_idx = np.argmax(similarities)
        most_relevant_resume_chunks.append({
            "job_chunk": job_chunks[i],
            "best_matching_resume_chunk": resume_chunks[best_idx],
            "score": similarities[best_idx]
        })
    return most_relevant_resume_chunks



# def get_resume_suggestions(llm, job_chunks, most_relevant_resume_chunks):
#     prompt_resume_suggestions = PromptTemplate.from_template(
#     """
# You are a career assistant helping someone tailor their resume for a specific job.

# Given the job description and parts of the resume most relevant to that job, suggest 3–5 specific improvements to their resume. Focus on making the resume better aligned with the role, especially in terms of responsibilities, keywords, and skills.

# Do NOT include any generic tips or advice.
# Do NOT include any markdown, headers, explanations, or commentary.
# Do NOT start with phrases like "Based on the job description..."
# Do NOT wrap the output in triple backticks.
# Do NOT include any code blocks.
# The suggestions should be specific about the keywords to be added or changed, the skills to highlight, and any responsibilities that should be emphasized or added.
# The suggestions should align with the job description provided. If the job description is not relevant to the resume chunk, explicitly
# state that the job description is not relevant to the resume chunk. 

# If the resume chunk is not relevant, explicitly say so in one of the suggestions.

# Your output must be a **valid JSON list of plain strings only**. Example:
# [
#   "Replace generic phrases like 'team player' with specific outcomes like 'collaborated with a team of 5 to improve processing time by 30%'",
#   "Mention Python and SQL if used in projects, as they are listed in the job description"
# ]

# ### JOB DESCRIPTION:
# {job_text}

# ### MATCHED RESUME CHUNKS:
# {resume_chunks}
# """
# )
#     chain_resume_suggestions = prompt_resume_suggestions | llm
#     resume_tips = chain_resume_suggestions.invoke({
#         "job_text": "\n\n".join([chunk.page_content for chunk in job_chunks]),
#         "resume_chunks": "\n\n".join([
#             item["best_matching_resume_chunk"].page_content for item in most_relevant_resume_chunks
#         ])
#     })
    
#     print(f"Raw response from model: {resume_tips.content}")
    
#     # Sanitize the response to remove array brackets and extra quotes
#     response = resume_tips.content.strip()
#     response = response.lstrip('[').rstrip(']')  # Remove array brackets

#     # Split based on double quotes, filter out empty strings
#     suggestions_list = [suggestion.strip() for suggestion in response.split('"') if suggestion.strip()]

#     return suggestions_list
def get_resume_suggestions(llm, job_chunks, most_relevant_resume_chunks):
    prompt_resume_suggestions = PromptTemplate.from_template(
    """
You are an expert resume coach. Your task is to help a candidate tailor their resume for a specific job.
# Given the job description and parts of the resume most relevant to that job, suggest 5-8 specific improvements to their resume. Focus on making the resume better aligned with the role, especially in terms of responsibilities, keywords, and skills.

You will be given:
- The job description (as text)
- Labeled chunks of the candidate's resume (each with [SECTION: ...] or [PROJECT: ...] or [EXPERIENCE: ...] labels).


**Rules:**
- When making suggestions, reference each section exactly as labeled (e.g., [SECTION: S&P Global Experience]).
- NEVER move or attribute achievements, tools, or skills from one section/project to another.
- Base every suggestion only on the *actual content* in the provided resume chunks. Do not invent or assume any experience not present.
- Do not use any specific phrases from the instruction examples. Instead, rephrase or rewrite based on the actual text in the resume.
- If the job description requests a skill or tool not mentioned in the resume, only suggest adding it if it is reasonable based on existing experience (e.g., suggest adding SQL only if similar data work is present).
- If you are unsure, do not make up or assume. Leave that skill/experience out.
- Do NOT delete degrees or valid internships/projects if the underlying skills are relevant, even if the title does not match exactly. Instead, suggest rewording for relevance.
- Use specific section names or quoted phrases from the resume whenever possible.
- Prefer suggestions that clarify or quantify impact, but do not invent numbers.
- If no relevant improvements are possible, return an empty list.
- Only refer to information present in the resume chunks provided (you may summarize, combine, or rephrase, but do not invent experiences or skills).
- If the job description mentions specific tools, methods, or results not in the resume, recommend only if they are plausibly related to the candidate’s actual experience.
- Prioritize suggestions that demonstrate quantifiable achievements or real impact (numbers, percentages, or clear outcomes, if available).
- You may recommend reorganizing or renaming sections, *but* do NOT recommend deleting valid education, internships, or projects if their skills/tools are relevant—even if the title doesn't exactly match the job.
- Instead of removing, suggest ways to reframe or reword for maximum relevance. If an internship or project is relevant but not an exact title match, **highlight how to rewrite it** to make the fit obvious.
- Do NOT suggest generic or template advice, such as “Add Python to your skills” if Python is already present.
- Do NOT suggest removing or rewording a degree (even if it is not mentioned in the job description).
- If the job description is inappropriate, not a professional job description, or is irrelevant to resumes (e.g., contains unrelated or unsafe content), reply with a single suggestion: "The job description appears invalid or unrelated to a professional context. Please provide a valid job description."
- If the job description is not relevant to the resume chunk, explicitly say so in one of the suggestions.


# Do NOT include any generic tips or advice.
# Do NOT include any markdown, headers, explanations, or commentary.
# Do NOT start with phrases like "Based on the job description..."
# Do NOT wrap the output in triple backticks.
# Do NOT include any code blocks.


**Return only a valid JSON list of plain strings.**

**Example (for format only):**
[
  "Modify the '<section>' to emphasize <relevant skills or achievements>",
  "Reword the '<project>' description to mention <technology> as used in the job description"
]

### JOB DESCRIPTION:
{job_text}

### RESUME CONTENT:
{resume_chunks}
"""
    )

    chain_resume_suggestions = prompt_resume_suggestions | llm
    resume_tips = chain_resume_suggestions.invoke({
        "job_text": "\n\n".join([chunk.page_content for chunk in job_chunks]),
        "resume_chunks": "\n\n".join([
            item["best_matching_resume_chunk"].page_content for item in most_relevant_resume_chunks
        ])
    })
    
    print(f"Raw response from model: {resume_tips.content}")

    # Use regex to extract each suggestion inside quotes (don't strip quotes first!)
    suggestions_list = re.findall(r'"(.*?)"', resume_tips.content)

    return suggestions_list
