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
import subprocess
import requests
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

def read_latex_resume(tex_path):
    with open(tex_path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf_resume(pdf_path):
    resume_loader = PyPDFLoader(pdf_path)
    resume_docs = resume_loader.load()
    return resume_docs[0].page_content

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


def get_resume_suggestions(llm, job_chunks, most_relevant_resume_chunks, is_latex=False):
    prompt_resume_suggestions = PromptTemplate.from_template(
    """
You are an expert resume coach. Your task is to help a candidate tailor their resume for a specific job.
Given the job description and parts of the resume most relevant to that job, suggest 5-8 specific improvements to their resume. Focus on making the resume better aligned with the role, especially in terms of responsibilities, keywords, and skills.

You will be given:
- The job description (as text)
- Chunks of the candidate's resume (plain text if PDF, LaTeX code if .tex).


**Instructions:**
- Suggest only the most direct, concise, and actionable changes to the resume.
- For each suggestion:
    - **Be specific:** Point out the exact line/section to modify, or provide the new line to add.
    - For LaTeX resumes: If rewording or adding, output the full LaTeX line and highlight new content with \\textbf{{}} or \\hl{{}} (assume \\usepackage{{soul}} is in the preamble)." if is_latex else ""
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

    suggestions_list = re.findall(r'"(.*?)"', resume_tips.content)
    return suggestions_list

def get_resume_suggestions_latex(llm, latex_code, job_description):
    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate.from_template(
        r"""
You are an expert resume coach and LaTeX document editor. Your task is to suggest precise, actionable improvements to a candidate's LaTeX resume so it better matches a given job description.
You are expert in ATS (Automated Tracking System) optimization for resumes, ensuring that the LaTeX code is not only well-structured but also contains the right keywords and skills to pass ATS filters.
You may give 5-8 specific suggestions to improve the resume as per ATS standards.

You will be given:
- The full job description as text.
- The candidate's LaTeX resume code as a single string (not parsed sections).

**Instructions:**
- Suggest **exact LaTeX code modifications** to make the resume stronger for this job.
- Each suggestion should be a clear, direct code change:
    - `"action": "replace"`: Replace a full line in the resume (show the old line and the new line).
    - `"action": "add"`: Insert a new LaTeX line (state after which line/section to add).
    - `"action": "remove"`: Remove a specific line (show the full line to be removed).
- For **additions**, use LaTeX commands like `\textbf{{...}}` or `\hl{{...}}` to highlight the new or important content (assume `\usepackage{{soul}}` is in the preamble).
- **Do NOT** invent or assume skills/experience; base all suggestions only on what is present in the LaTeX code and job description.
- **Do NOT** suggest the same keyword/skill multiple times if it already appears anywhere in the code.
- **Do NOT** give generic tips or commentary—just the direct changes.
- **Do NOT** suggest changes like renaming education to academic background or similar; focus on specific content changes. 
- Do NOT give generic tips—just give the **actual LaTeX code to change or add**.
- If the job description is inappropriate or unrelated, respond with a single suggestion: `[{{"action": "invalid", "reason": "Job description appears invalid or unrelated to a professional context."}}]`

Produce **one numbered list** of **plain-text** suggestions, in this exact format:

1. **Replace** the line  
   old: `<old_line>`  
   new: `<new_line>`

2. **Add** after `<after_line>`:  
   `<new_line>`

3. **Remove** the line:  
   `<old_line>`

– and so on for every suggestion.

If no changes are needed, just output:

`1. No changes needed — your LaTeX is already aligned.`

### JOB DESCRIPTION:
{job_text}

### RESUME LATEX CODE:
{resume_code}
"""
    )
    chain = prompt | llm
    result = chain.invoke({
        "job_text": job_description,
        "resume_code": latex_code,
    })
    print(result.content)


    raw_text = result.content.strip()

    # Split into blocks: from "N. " to just before the next "<number>."
    pattern = r'(\d+\.\s[\s\S]*?)(?=(?:\n\d+\.|\Z))'
    blocks = re.findall(pattern, raw_text, flags=re.MULTILINE)

    return [blk.strip() for blk in blocks]


def extract_suggestion_action(suggestion):
    """Extract action, section, old, and new from a suggestion string."""
    # Replace pattern
    m = re.match(r"In \[SECTION: (.*?)\], replace '(.*?)' with '(.*?)'\.", suggestion)
    if m:
        return {'action': 'replace', 'section': m.group(1), 'old': m.group(2), 'new': m.group(3)}
    # Add pattern
    m = re.match(r"Add the following line to \[SECTION: (.*?)\]: '(.*?)'\.", suggestion)
    if m:
        return {'action': 'add', 'section': m.group(1), 'new': m.group(2)}
    # Remove pattern
    m = re.match(r"In \[SECTION: (.*?)\], remove the line '(.*?)'\.", suggestion)
    if m:
        return {'action': 'remove', 'section': m.group(1), 'old': m.group(2)}
    return None

def get_section_indices(lines, section):
    """Find start and end indices for a LaTeX section."""
    start = None
    end = None
    for idx, line in enumerate(lines):
        if line.strip().startswith(r"\section") and section in line:
            start = idx
            break
    if start is not None:
        # End at next \section or \end{document}
        for idx2 in range(start+1, len(lines)):
            if lines[idx2].strip().startswith(r"\section") or lines[idx2].strip() == r"\end{document}":
                end = idx2
                break
        if end is None:
            end = len(lines)
    return start, end


# def parse_suggestion(suggestion):
#     match = re.search(r"replace '(.*?)' with '(.*?)'", suggestion)
#     if match:
#         return match.group(1), match.group(2)
#     return None, None



def apply_suggestions_to_latex(latex_code, suggestions, st=None):
    lines = latex_code.splitlines()
    approved_actions = []
    for suggestion in suggestions:
        parsed = extract_suggestion_action(suggestion)
        if not parsed:
            continue
        action = parsed['action']
        section = parsed.get('section')
        old = parsed.get('old')
        new = parsed.get('new')
        section_start, section_end = get_section_indices(lines, section) if section else (None, None)

        # Streamlit interactive approval
        if st:
            msg = ""
            if action == "replace":
                msg = f"**[Section: {section}]**\n\nReplace:\n\n`{old}`\n\nwith\n\n`{new}`"
            elif action == "add":
                msg = f"**[Section: {section}]**\n\nAdd line:\n\n`{new}`"
            elif action == "remove":
                msg = f"**[Section: {section}]**\n\nRemove line:\n\n`{old}`"
            approve = st.radio(msg, ("Approve", "Skip"), key=f"{action}_{section}_{old}_{new}")
        else:
            approve = "Approve"  # CLI fallback

        if approve == "Approve":
            # Apply change to the lines (don't apply yet; just record)
            approved_actions.append(parsed)

    # Now, after all approvals, apply changes in batch (to avoid index issues)
    final_code = "\n".join(lines)
    final_code = apply_approved_latex_changes(final_code, approved_actions)
    return final_code


import re

def apply_approved_latex_changes(tex_code: str, approved_actions: list) -> str:
    """
    Apply a list of approved actions (replace/add/remove) to the LaTeX source.
    approved_actions is a list of dicts, each with:
      - action: "replace" | "add" | "remove"
      - old:    (for replace/remove) the exact substring to replace/remove
      - new:    (for replace/add) the new text to insert
      - section:(for add) the section name into whose header we insert
    """
    for act in approved_actions:
        if act["action"] == "replace":
            # simple global replace of old → new
            tex_code = tex_code.replace(act["old"], act["new"])

        elif act["action"] == "remove":
            # remove the exact old substring
            tex_code = tex_code.replace(act["old"], "")

        elif act["action"] == "add":
            # Build a regex to find the \section{SectionName} line
            # and insert new text right after it.
            # We avoid bad \e escapes by concatenating raw strings.
            section_name = act["section"]
            new_text = act["new"]

            # pattern: literal '\section' (optional *), then '{SectionName}'
            pattern = r"(\\section\*?\{" + re.escape(act["section"]) + r"\})"
            repl    = r"\1\n" + act["new"]
            # perform a single substitution
            tex_code = re.sub(pattern, repl, tex_code, count=1)

    return tex_code
