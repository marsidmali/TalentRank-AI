import streamlit as st
from src.resume_parser import ResumeParser
from src.ranker import Ranker
from src.utils.config import DEFAULT_WEIGHTS

if 'weights' not in st.session_state:
    st.session_state.weights = DEFAULT_WEIGHTS.copy()
if 'reset_trigger' not in st.session_state:
    st.session_state.reset_trigger = 0
if 'extra_skills' not in st.session_state:
    st.session_state.extra_skills = ""

# Title and Text Input for Job Description
st.title("HR Candidate Ranking Interface")
st.write("Provide a job description and upload candidate resumes to rank candidates based on customizable criteria.")
st.subheader("Enter Job Description")
job_description = st.text_area("Enter the job description below:")

# Candidate Resumes Upload
st.subheader("Upload Candidate Resumes")
candidate_files = st.file_uploader("Upload Candidate Resumes (PDFs)", type="pdf", accept_multiple_files=True)

# Sidebar for adjustable weights
def create_weight_callback(weight_key):
    def callback():
        st.session_state.weights[weight_key] = st.session_state[f"slider_{weight_key}"]
    return callback

st.sidebar.header("Adjustable Ranking Weights")
for key in DEFAULT_WEIGHTS:
    st.sidebar.slider(
        key,
        min_value=-1.0 if "Missing" in key else 0.0,
        max_value=0.0 if "Missing" in key else 1.0,
        value=st.session_state.weights[key],
        key=f"slider_{key}",
        on_change=create_weight_callback(key)
    )

# Input box for extra skills
extra_skills = st.sidebar.text_input(
    "Prompt extra skills (comma-separated)",
    value=st.session_state.extra_skills,
    key=f"extra_skills_{st.session_state.reset_trigger}"
)
st.session_state.extra_skills = extra_skills

# Reset and Confirm buttons
col1, col2 = st.sidebar.columns(2)
if col1.button("Reset Weights", key=f"reset_{st.session_state.reset_trigger}"):
    st.session_state.weights = DEFAULT_WEIGHTS.copy()
    st.session_state.reset_trigger += 1
    st.experimental_rerun()

if col2.button("Confirm", key=f"confirm_{st.session_state.reset_trigger}"):
    st.sidebar.write("Weights confirmed.")

# Rank Candidates Button
if st.button("Rank Candidates"):
    if job_description.strip() and candidate_files:
        candidate_data = {candidate.name: candidate.read() for candidate in candidate_files}
        parser = ResumeParser(
            candidate_data=candidate_data,
        )
        df = parser.main()
        result = {}
        for _, row in df.iterrows():
            filename = row['filename']
            header = row['header']
            section_text = row['section text']
            if filename not in result:
                result[filename] = {}
            result[filename][header] = section_text

        ranker = Ranker(
            job_listing=job_description,
            weights=st.session_state.weights,
            extra_skills=st.session_state.extra_skills
        )

        rankings = ranker.evaluate_candidates(result)

        # Display rankings
        st.subheader("Candidate Rankings")
        for rank, (filename, scores) in enumerate(rankings.items(), 1):
            final_score = scores['Final Score']
            st.markdown(f"{rank}. [{filename}](Resumes/{filename}) - Score: {final_score}")

    else:
        st.warning("Please provide a job description and upload candidate resumes.")
