import os
os.environ["STREAMLIT_FILE_WATCHER_TYPE"] = "none"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag, ne_chunk

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True)

stop_words = set(stopwords.words('english'))

# Helper functions
def extract_text_from_pdf(pdf_file):
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error("Invalid PDF file. Please upload a valid PDF.")
        return ""

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

def calculate_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

def extract_skills(text):
    try:
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        tree = ne_chunk(pos_tags)
        skills = []
        current_skill = []
        for subtree in tree:
            if isinstance(subtree, nltk.Tree):
                continue
            if isinstance(subtree, tuple):
                word, pos = subtree
                if pos.startswith('NN'):
                    current_skill.append(word)
                else:
                    if current_skill:
                        skills.append(' '.join(current_skill))
                        current_skill = []
        if current_skill:
            skills.append(' '.join(current_skill))
        skills = [skill.lower() for skill in skills if len(skill) > 1 and skill not in stop_words]
        return list(set(skills))
    except Exception as e:
        st.error(f"Error extracting skills: {str(e)}. Please ensure NLTK resources are downloaded.")
        return []

def is_skill_in_text(skill, text):
    text = text.lower()
    skill_pattern = r'\b' + re.escape(skill) + r'\b'
    return re.search(skill_pattern, text) is not None

# Load model once
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

def determine_fit_status(similarity_score, missing_skills_count, total_skills):
    if similarity_score > 0.75 and missing_skills_count == 0:
        return "Excellent match! Your resume aligns well and includes all required skills."
    elif similarity_score > 0.75 and missing_skills_count > 2:
        return "Even though your resume aligns well semantically, several required skills are missing. Consider adding these skills to improve your fit."
    elif 0.5 < similarity_score <= 0.75 and missing_skills_count <= 1:
        return "Good alignment with minimal skill gaps. Adding the missing skill(s) could strengthen your resume."
    elif 0.5 < similarity_score <= 0.75:
        return "Fair match. Your resume has some alignment with the job description."
    elif similarity_score <= 0.5 and missing_skills_count in [2, 3]:
        return "You may need to improve your skills and also your resume."
    elif similarity_score <= 0.5 and missing_skills_count == total_skills:
        return "Poor match. Your resume lacks both semantic alignment and required skills. Tailor your resume and add relevant skills."
    else:
        return "Low match. Consider tailoring your resume to better align with the job description."

def show_recruiter_mode():
    st.header("Recruiter Mode: Bulk Resume Matching & Ranking")

    job_description = st.text_area(
        "Enter Job Description",
        height=200,
        placeholder="Paste your job description here..."
    )

    if job_description:
        with st.spinner("Extracting skills..."):
            skills = extract_skills(job_description)
        st.subheader("Extracted Skills from Job Description")
        if 'skills' not in st.session_state:
            st.session_state.skills = skills
        edited_skills = st.text_area(
            "Edit or Add Skills (one per line)",
            value='\n'.join(st.session_state.skills),
            height=150
        )
        if st.button("Finalize Skills"):
            st.session_state.skills = [skill.strip().lower() for skill in edited_skills.split('\n') if skill.strip()]
            st.success("Skills finalized!")

    uploaded_files = st.file_uploader(
        "Upload Resume PDFs (multiple)",
        type=['pdf'],
        accept_multiple_files=True
    )

    if job_description and uploaded_files and 'skills' in st.session_state:
        with st.spinner("Processing resumes..."):
            job_clean = clean_text(job_description)
            job_embedding = model.encode([job_clean])[0]
            results = []

            for uploaded_file in uploaded_files:
                resume_text = extract_text_from_pdf(uploaded_file)
                if not resume_text:
                    continue
                resume_clean = clean_text(resume_text)
                resume_embedding = model.encode([resume_clean])[0]
                score = calculate_similarity(resume_embedding, job_embedding)
                score = min(score, 0.9999)

                matched_skills = [skill for skill in st.session_state.skills if is_skill_in_text(skill, resume_text)]
                missing_skills = [skill for skill in st.session_state.skills if not is_skill_in_text(skill, resume_text)]
                fit_status = determine_fit_status(score, len(missing_skills), len(st.session_state.skills))

                results.append({
                    "filename": uploaded_file.name,
                    "score": score,
                    "fit_status": fit_status,
                    "matched_skills": matched_skills,
                    "missing_skills": missing_skills,
                    "resume_text": resume_text
                })

            df_results = pd.DataFrame(results).sort_values(by='score', ascending=False)

        st.subheader("Candidate Ranking")
        for idx, row in df_results.iterrows():
            st.markdown(f"### {row['filename']}")
            if "Excellent match" in row['fit_status'] or "Good alignment" in row['fit_status']:
                st.success(f"**Fit Status:** {row['fit_status']}")
            elif "Fair match" in row['fit_status']:
                st.info(f"**Fit Status:** {row['fit_status']}")
            else:
                st.warning(f"**Fit Status:** {row['fit_status']}")
            st.write(f"**Required Skills:** {', '.join(st.session_state.skills)}")
            st.write(f"**Matched Skills:** {', '.join(row['matched_skills']) if row['matched_skills'] else 'None'}")
            st.write(f"**Missing Skills:** {', '.join(row['missing_skills']) if row['missing_skills'] else 'None'}")
            with st.expander("View Extracted Resume Text"):
                st.write(row['resume_text'])

        st.subheader("Similarity Score Distribution (Internal)")
        plt.figure(figsize=(8,4))
        plt.hist(df_results['score'], bins=10, color='skyblue', edgecolor='black')
        plt.xlabel('Similarity Score')
        plt.ylabel('Number of Resumes')
        plt.title('Resume Similarity Scores Distribution')
        st.pyplot(plt)

        st.markdown("**Note:** This assessment is based on semantic similarity and skill matching. Actual fit may vary depending on other factors.")
    else:
        st.info("Please enter a job description, upload at least one resume PDF, and finalize skills to see results.")

def show_candidate_mode():
    st.header("Candidate Mode: Personalized Resume Feedback")

    job_description = st.text_area(
        "Paste or type the Job Description here:",
        height=200,
        placeholder="Enter the full job description text..."
    )

    if job_description:
        with st.spinner("Extracting skills..."):
            skills = extract_skills(job_description)
        st.subheader("Extracted Skills from Job Description")
        if 'skills' not in st.session_state:
            st.session_state.skills = skills
        edited_skills = st.text_area(
            "Edit or Add Skills (one per line)",
            value='\n'.join(st.session_state.skills),
            height=150
        )
        if st.button("Finalize Skills"):
            st.session_state.skills = [skill.strip().lower() for skill in edited_skills.split('\n') if skill.strip()]
            st.success("Skills finalized!")

    resume_file = st.file_uploader("Upload your resume PDF:", type=['pdf'])

    if job_description and resume_file and 'skills' in st.session_state:
        with st.spinner("Processing resume..."):
            resume_text = extract_text_from_pdf(resume_file)
            if not resume_text:
                return
            resume_embedding = model.encode([clean_text(resume_text)])[0]
            job_embedding = model.encode([clean_text(job_description)])[0]
            similarity_score = calculate_similarity(resume_embedding, job_embedding)
            similarity_score = min(similarity_score, 0.9999)

            matched_skills = [skill for skill in st.session_state.skills if is_skill_in_text(skill, resume_text)]
            missing_skills = [skill for skill in st.session_state.skills if not is_skill_in_text(skill, resume_text)]
            fit_status = determine_fit_status(similarity_score, len(missing_skills), len(st.session_state.skills))

        st.subheader("Resume Analysis")
        if "Excellent match" in fit_status or "Good alignment" in fit_status:
            st.success(f"**Fit Status:** {fit_status}")
        elif "Fair match" in fit_status:
            st.info(f"**Fit Status:** {fit_status}")
        else:
            st.warning(f"**Fit Status:** {fit_status}")

        st.write(f"**Required Skills:** {', '.join(st.session_state.skills)}")
        st.write(f"**Matched Skills:** {', '.join(matched_skills) if matched_skills else 'None'}")
        st.write(f"**Missing Skills:** {', '.join(missing_skills) if missing_skills else 'None'}")

        total_skills = len(st.session_state.skills)
        matched_count = len(matched_skills)
        if total_skills > 0:
            progress = matched_count / total_skills
            st.progress(progress)
            st.write(f"**Skills Matched:** {matched_count} / {total_skills}")

        with st.expander("View Extracted Resume Text"):
            st.write(resume_text)

        st.markdown("**Note:** This assessment is based on semantic similarity and skill matching. Actual fit may vary depending on other factors.")
    else:
        st.info("Please enter a job description, upload your resume PDF, and finalize skills to get feedback.")

def main():
    st.set_page_config(page_title="SmartMatch Resume", layout="wide")
    st.title("SmartMatch Resume")
    st.markdown("""
    **An intelligent platform that semantically matches resumes with job descriptions,  
    providing detailed insights and actionable feedback for recruiters and candidates alike.**
    """)

    mode = st.sidebar.radio("Select Mode:", ["Recruiter Mode", "Candidate Mode"])

    if mode == "Recruiter Mode":
        show_recruiter_mode()
    else:
        show_candidate_mode()

if __name__ == "__main__":
    main()