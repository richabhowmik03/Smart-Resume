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


from utils import (
    load_env, get_llm, read_latex_resume, load_pdf_resume, load_job_posting,
    extract_job_postings, split_texts, get_embedding_model,
    create_vectorstores, get_most_relevant_resume_chunks,
    get_resume_suggestions, get_resume_suggestions_latex, 
    extract_suggestion_action, get_section_indices, 
    apply_suggestions_to_latex, apply_approved_latex_changes
)

if "review_index" not in st.session_state:
    st.session_state.review_index = 0
if "approved_changes" not in st.session_state:
    st.session_state.approved_changes = []
if "tex_code" not in st.session_state:
    st.session_state.tex_code = None
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []


load_env()
llm = get_llm()

st.title("🧠 Smart Resume Tailor with Groq + LangChain")

st.markdown(
    """
    <style>
        .centered { display: flex; align-items: center; }
        .upload-col { min-width: 210px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Resume file uploader
resume_type = st.radio(
    "How do you want to upload your resume?",
    ("PDF", "LaTeX (.tex)"),
    help="LaTeX resumes allow you to download the AI-edited file for manual review or recompilation."
)

if resume_type == "PDF":
    pdf_file = st.file_uploader(
        "📄 Upload your resume (PDF)",
        type=["pdf"],
        key="upload_pdf"
    )
elif resume_type == "LaTeX (.tex)":
    latex_file = st.file_uploader(
        "📄 Upload your resume (LaTeX .tex file)",
        type=["tex"],
        key="upload_tex"
    )

# pdf_file = st.file_uploader("📄 Upload your resume (PDF)", type=["pdf"])

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



# -------------- PDF Resume Logic -------------------
if resume_type == "PDF" and pdf_file and job_description and st.button("🔍 Analyze Resume"):
    with st.spinner("Loading resume and analyzing..."):
        with open("temp_resume.pdf", "wb") as f:
            f.write(pdf_file.read())
        resume_docs = load_pdf_resume("temp_resume.pdf")
        resume_text = load_pdf_resume("temp_resume.pdf")
        resume_chunks, job_chunks = split_texts(resume_text, job_description)
        embed_model = get_embedding_model()
        resume_db, _ = create_vectorstores(resume_chunks, job_chunks, embed_model)
        similarity_results = get_most_relevant_resume_chunks(
            job_chunks, resume_chunks,
            job_embeddings=embed_model.embed_documents([doc.page_content for doc in job_chunks]),
            resume_embeddings=embed_model.embed_documents([doc.page_content for doc in resume_chunks])
        )
        suggestions = get_resume_suggestions(llm, job_chunks, similarity_results)

    if suggestions:
        st.success("🎯 Done! Here are your resume suggestions:")
        if isinstance(suggestions[0], str):
            for suggestion in suggestions:
                st.write(f"- {suggestion}")
        elif isinstance(suggestions[0], dict):
            for s in suggestions:
                st.json(s)
        else:
            st.write(suggestions)
    else:
        st.info("No suggestions found.")

    with st.expander("See extracted job description"):
        st.write(job_description)
    with st.expander("See uploaded resume text"):
        st.write(resume_text)

# --- LaTeX Resume Logic (refactored) ---
if resume_type == "LaTeX (.tex)":
    import re

    # Load uploaded LaTeX into session state on every upload
    if latex_file is not None:
        st.session_state.tex_code = latex_file.read().decode("utf-8")

    # Display raw LaTeX source or prompt to upload
    if st.session_state.get("tex_code"):
        st.subheader("📜 LaTeX Source")
        st.code(st.session_state.tex_code, language="latex")
    else:
        st.info("Upload a .tex file to see its source here.")

    # Ensure resume (.tex) and job description are present
    has_tex = bool(st.session_state.get("tex_code"))
    has_jd = bool(job_description and job_description.strip())

    # Analyze button
    if has_tex and has_jd:
        if st.button("🔍 Analyze Resume (LaTeX)", key="analyze_latex"):
            with st.spinner("Analyzing LaTeX resume…"):
                
                st.session_state.raw_suggestions = get_resume_suggestions_latex(
                    llm,
                    st.session_state.tex_code,
                    job_description
                )
                st.session_state.review_index = 0
                st.session_state.approved_changes = []
                
                st.write("**Suggestions retrieved:**")
                for suggestion in st.session_state.raw_suggestions:
                    st.text(suggestion)
    else:
        if not has_jd:
            st.warning("Please provide a job description before analyzing.")
        if not has_tex:
            st.warning("Please upload your LaTeX resume file (.tex) to analyze.")

    # Interactive review of suggestions
    suggestions = st.session_state.get("raw_suggestions", [])
    idx = st.session_state.get("review_index", 0)

    if suggestions:
        if idx < len(suggestions):
            raw = suggestions[idx]
            st.markdown(f"### Suggestion {idx+1} of {len(suggestions)}")
            st.text(raw)

            # Determine action
            action = None
            if "**Replace**" in raw:
                action = "replace"
            elif "**Add**" in raw:
                action = "add"
            elif "**Remove**" in raw:
                action = "remove"

            # Initialize parsed fields
            old_line = None
            new_line = None
            section = None

            # Extract old_line for replace/remove
            if action in ("replace", "remove"):
    
                m = re.search(r"old:\s*`([^`]+)`", raw)
                if not m:
                    # fallback for Remove-only blocks: grab the first back-quoted line
                    m = re.search(r"`([^`]+)`", raw)
                if m:
                    old_line = m.group(1)
                else:
                    st.warning("Could not parse the original line from this suggestion.")

            # Extract new_line or section
            if action == "replace":
                m = re.search(r"new:\s*`([^`]+)`", raw)
                if m:
                    new_line = m.group(1)
                else:
                    st.warning("Could not parse the new line from this suggestion.")
            elif action == "add":
                # after_line gives section header
                m = re.search(r"after\s*`([^`]+)`", raw, re.IGNORECASE)
                if m:
                    after_line = m.group(1)
                    # extract section name interior
                    m_sec = re.match(r"\\section\*?\{(.+?)\}", after_line)
                    if m_sec:
                        section = m_sec.group(1)
                else:
                    st.warning("Could not parse where to add from this suggestion.")
                # new_line: first standalone backtick content
                m2 = re.search(r"\n\s*`([^`]+)`", raw)
                if m2:
                    new_line = m2.group(1)
                else:
                    st.warning("Could not parse the added line from this suggestion.")

            # Build parsed suggestion
            parsed = {"action": action}
            valid = True
            if action == "replace":
                if old_line and new_line:
                    parsed.update({"old": old_line, "new": new_line})
                else:
                    valid = False
            elif action == "add":
                if new_line and section:
                    parsed.update({"new": new_line, "section": section})
                else:
                    valid = False
            elif action == "remove":
                if old_line:
                    parsed.update({"old": old_line})
                else:
                    valid = False

            if not valid:
                st.error("Skipping this suggestion because it could not be fully parsed.")
            else:
                col1, col2 = st.columns(2)
                if col1.button("✅ Approve", key=f"approve_{idx}"):
                    st.session_state.approved_changes.append(parsed)
                    st.session_state.review_index += 1
                if col2.button("❌ Reject", key=f"reject_{idx}"):
                    st.session_state.review_index += 1
        else:
            # All suggestions processed → apply changes
            if st.session_state.get("tex_code"):
                actions = []
                for act in st.session_state.approved_changes:
                    # Only include well-formed actions
                    if act["action"] in ("replace", "remove") and not act.get("old"):
                        continue
                    if act["action"] == "add" and not act.get("section"):
                        continue
                    actions.append(act)

                # Apply approved actions
                final_tex = st.session_state.tex_code
                for act in actions:
                    if act["action"] == "replace":
                        final_tex = final_tex.replace(act["old"], act["new"])
                    elif act["action"] == "remove":
                        final_tex = final_tex.replace(act["old"], "")
                    elif act["action"] == "add":
                        # Insert using a function to avoid bad escape issues
                        section_name = act["section"]
                        new_text = act["new"]
                        pattern = r"(\\section\*?\{" + re.escape(section_name) + r"\})"
                        final_tex = re.sub(pattern,
                                           lambda m: m.group(1) + "\n" + new_text,
                                           final_tex,
                                           count=1)
                st.success("🎉 All suggestions processed! Download your updated LaTeX file below.")
                st.download_button(
                    "⬇️ Download updated .tex",
                    data=final_tex,
                    file_name="resume_updated.tex",
                    mime="text/plain"
                )
                if st.button("🔄 Start Over"):
                    for key in ("tex_code", "raw_suggestions", "review_index", "approved_changes"):
                        st.session_state.pop(key, None)
            else:
                st.error("No LaTeX source found—please re-upload your .tex and try again.")

# if resume_file and job_description and st.button("🔍 Analyze Resume"):
#     with st.spinner("Loading resume and analyzing..."):
#         # Load and process resume
#         with open("temp_resume.pdf", "wb") as f:
#             f.write(resume_file.read())
#         st.write("Checkpoint 1: Resume PDF saved!")
#         resume_docs = load_resume("temp_resume.pdf")
#         st.write("Checkpoint 2: Resume PDF loaded into docs!")
#         resume_text = resume_docs[0].page_content

#         # Split both texts
#         resume_chunks, job_chunks = split_texts(resume_text, job_description)
#         st.write("Checkpoint 3: Resume/job split")

#         # Embed and create vectorstores
#         embed_model = get_embedding_model()
#         st.write("Checkpoint 4: Embedding model loaded")
#         resume_db, _ = create_vectorstores(resume_chunks, job_chunks, embed_model)
#         st.write("Checkpoint 5: Embeddings created")


#         # Find most relevant resume chunks
#         similarity_results = get_most_relevant_resume_chunks(
#             job_chunks, resume_chunks,
#             job_embeddings=embed_model.embed_documents([doc.page_content for doc in job_chunks]),
#             resume_embeddings=embed_model.embed_documents([doc.page_content for doc in resume_chunks])
#         )

#         # Get resume improvement suggestions
#         suggestions = get_resume_suggestions(llm, job_chunks, similarity_results)

#     st.success("🎯 Done! Here are your resume suggestions:")
#     for suggestion in suggestions:
#         st.write(f"- {suggestion}")

#     with st.expander("See extracted job description"):
#         st.write(job_description)

#     with st.expander("See uploaded resume text"):
#         st.write(resume_text)
