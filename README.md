# 🔍 SmartMatch Resume

**SmartMatch Resume** is an intelligent resume screening and analysis tool that semantically matches resumes with job descriptions using NLP and deep learning. It supports two modes:

- 🧑‍💼 **Recruiter Mode**: Upload and rank multiple candidate resumes for a given job description.
- 👨‍💻 **Candidate Mode**: Upload a resume and get personalized feedback for a specific job role.

---

## 🚀 Features

- ✅ Semantic similarity scoring using `SentenceTransformer`
- ✅ Skill extraction via NLP (`nltk`)
- ✅ PDF text extraction using `pdfplumber`
- ✅ Resume ranking for recruiters
- ✅ Detailed skill match analysis and feedback for candidates
- ✅ Interactive visualizations and UI via Streamlit

---

## 🛠 Installation

```bash
# Clone the repo
git clone https://github.com/jkplearner/smartMatchResumes.git
cd smartMatchResumes

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
