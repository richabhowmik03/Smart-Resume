import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st

# Add a toggle button
# dark_mode = st.toggle("🌙 Dark Mode", value=False)

# dark_css = """
# <style>
# .stApp {
#     background-color: #181818 !important;
#     color: #f0f0f0 !important;
# }
# .st-emotion-cache-1v0mbdj, .st-emotion-cache-10trblm, .st-emotion-cache-16txtl3, .st-emotion-cache-1c7y2kd {
#     color: #f0f0f0 !important;
# }
# h1, h2, h3, h4, h5, h6, label, .st-bx, .st-bu, .stText, .stMarkdown, .stTextInput, .stTextArea, .stDataFrame {
#     color: #f0f0f0 !important;
# }
# .stTextInput>div>div>input, .stTextArea>div>textarea {
#     background: #222 !important;
#     color: #eee !important;
# }
# .stButton>button {
#     background-color: #333 !important;
#     color: #eee !important;
#     border: 1px solid #444 !important;
# }
# </style>
# """

# light_css = """
# <style>
# .stApp {
#     background-color: #f9f9f9 !important;
#     color: #1a1a1a !important;
# }
# .st-emotion-cache-1v0mbdj, .st-emotion-cache-10trblm, .st-emotion-cache-16txtl3, .st-emotion-cache-1c7y2kd {
#     color: #1a1a1a !important;
# }
# h1, h2, h3, h4, h5, h6, label, .st-bx, .st-bu, .stText, .stMarkdown, .stTextInput, .stTextArea, .stDataFrame {
#     color: #1a1a1a !important;
# }
# .stTextInput>div>div>input, .stTextArea>div>textarea {
#     background: #fff !important;
#     color: #1a1a1a !important;
# }
# .stButton>button {
#     background-color: #e0e0e0 !important;
#     color: #111 !important;
#     border: 1px solid #aaa !important;
# }
# </style>
# """

# # Inject CSS based on toggle
# if dark_mode:
#     st.markdown(dark_css, unsafe_allow_html=True)
# else:
#     st.markdown(light_css, unsafe_allow_html=True)


import streamlit as st
from utils import (
    load_env, get_llm, load_resume, load_job_posting,
    extract_job_postings, split_texts, get_embedding_model,
    create_vectorstores, get_most_relevant_resume_chunks,
    get_resume_suggestions
)

load_env()
llm = get_llm()

st.title("🧠 Smart Resume Tailor with Groq + LangChain")

pdf_file = st.file_uploader("📄 Upload your resume (PDF)", type=["pdf"])

job_input_type = st.radio("How do you want to provide the job description?", ["Link", "Text"])

job_description = ""
if job_input_type == "Link":
    st.warning(
        "⚠️ **Disclaimer:** When using a job link, results may vary depending on the website's permissions and anti-scraping measures. "
        "Some sites may block scraping, which could lead to incomplete or missing job descriptions. "
        "For best results, consider copy-pasting the job description text."
    )

    job_url = st.text_input("Paste the job posting URL:")
    if job_url:
        from langchain_community.document_loaders import WebBaseLoader
        loader = WebBaseLoader(job_url)
        job_docs = loader.load()
        job_description = job_docs[0].page_content
elif job_input_type == "Text":
    job_description = st.text_area("Paste the job description here:")

if pdf_file and job_description and st.button("🔍 Analyze Resume"):
    with st.spinner("Loading resume and analyzing..."):
        # Load and process resume
        with open("temp_resume.pdf", "wb") as f:
            f.write(pdf_file.read())
        st.write("Checkpoint 1: Resume PDF saved!")
        resume_docs = load_resume("temp_resume.pdf")
        st.write("Checkpoint 2: Resume PDF loaded into docs!")
        resume_text = resume_docs[0].page_content

        # Split both texts
        resume_chunks, job_chunks = split_texts(resume_text, job_description)
        st.write("Checkpoint 3: Resume/job split")

        # Embed and create vectorstores
        embed_model = get_embedding_model()
        st.write("Checkpoint 4: Embedding model loaded")
        resume_db, _ = create_vectorstores(resume_chunks, job_chunks, embed_model)
        st.write("Checkpoint 5: Embeddings created")


        # Find most relevant resume chunks
        similarity_results = get_most_relevant_resume_chunks(
            job_chunks, resume_chunks,
            job_embeddings=embed_model.embed_documents([doc.page_content for doc in job_chunks]),
            resume_embeddings=embed_model.embed_documents([doc.page_content for doc in resume_chunks])
        )

        # Get resume improvement suggestions
        suggestions = get_resume_suggestions(llm, job_chunks, similarity_results)

    st.success("🎯 Done! Here are your resume suggestions:")
    for suggestion in suggestions:
        st.write(f"- {suggestion}")

    with st.expander("See extracted job description"):
        st.write(job_description)

    with st.expander("See uploaded resume text"):
        st.write(resume_text)
