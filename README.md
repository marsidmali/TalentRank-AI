# 📑TalentRank AI: Smart Resume Screening System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent resume ranking system powered by GPT-4o that automatically analyzes and ranks resumes based on job descriptions. Save hours of manual screening and find the best candidates faster! 🚀

## ✨ Features

- 📄 **Smart PDF Parsing**: Automatically extracts and structures information from PDF resumes
- 🎯 **Intelligent Matching**: Uses GPT-4o to understand and match candidate qualifications with job requirements
- 💡 **Customizable Scoring**: Flexible weighting system for different criteria:
  - Skills matching
  - Experience relevance
  - Education alignment
  - Job history analysis

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/marsidmali/talentrank-ai.git
cd talentrank-ai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to the `.env` file

## 🚀 Usage

1. Start the application:
```bash
streamlit run src/main.py
```

2. Upload resumes and job description
3. Adjust scoring weights (optional)
4. Get ranked results and scores

## 📁 Project Structure

```plaintext
talentrank-ai/
│
├── src/                  # Source code
│   ├── main.py          # Main application
│   ├── resume_parser.py # Resume parsing logic
│   ├── ranker.py       # Ranking algorithm
│   └── utils/          # Utilities
|
├── Resumes/            # Resume storage
├── requirements.txt    # Dependencies
└── README.md          # Documentation
```

## ⚙️ Configuration

The system uses the following default weights which can be customized:

```python
DEFAULT_WEIGHTS = {
    "Matching skills weight": 0.3,
    "Missing skills weight": -0.2,
    "Relevant job list weight": 0.2,
    "Relevant degree list weight": 0.1,
    "Years of relevant experience weight": 0.4
}
```

## 🔑 Environment Variables

Create a `.env` file with:
```plaintext
OPENAI_API_KEY=your_api_key_here
```


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

