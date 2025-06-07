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
# from langchain_community.vectorstores import Chroma  # Updated import
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

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
    resume_db = FAISS.from_documents(resume_chunks, embedding_model)
    job_db = FAISS.from_documents(job_chunks, embedding_model)
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


def get_resume_suggestions(llm, job_chunks, most_relevant_resume_chunks):
    prompt_resume_suggestions = PromptTemplate.from_template(
    """
You are an expert resume coach. Your task is to help a candidate tailor their resume for a specific job.

You will be given:
- The job description (as text)
- Labeled chunks of the candidate's resume (each with [SECTION: ...], [PROJECT: ...], or [EXPERIENCE: ...] labels).

**Instructions:**
- Suggest only the most direct, concise, and actionable changes to the resume.
- For each suggestion:
    - **Be specific:** Point out the exact line/section to modify, or provide the new line to add.
    - **Do NOT recommend adding a keyword or skill if it already appears anywhere in the resume, even if not in the ideal section.**
    - **If rewording is needed, show exactly what to replace and what to change it to.**
    - **If adding a new line, provide the full sentence as it should appear in the resume.**
    - Do NOT invent or assume experience, skills, or numbers that are not in the actual content.
    - Do NOT ask to add the same keyword to multiple sections.
    - Do NOT clutter the suggestions: Each suggestion should be a single, clear instruction (one per bullet).
    - Do NOT give generic tips or commentary—just the direct changes.

- If no relevant improvements are possible, return an empty list.
- If the job description is inappropriate or unrelated to professional work, return: ["The job description appears invalid or unrelated to a professional context. Please provide a valid job description."]

**Return only a valid JSON list of plain strings.**

**Examples:**
[
  "In [SECTION: Professional Summary], replace 'analyzing data' with '**building predictive models using Python and SQL**'.",
  "Add the following line to [TECHNICAL SKILLS]: '**AWS SageMaker, Model Validation**'.",
  "In [PROJECT: S&P Global], change 'forecasting sales' to 'forecasting sales using time series modeling'."
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

    import re
    suggestions_list = re.findall(r'"(.*?)"', resume_tips.content)
    return suggestions_list
