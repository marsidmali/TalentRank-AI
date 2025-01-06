import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Default weights configuration
DEFAULT_WEIGHTS = {
    "Matching skills weight": 0.3,
    "Missing skills weight": -0.2,
    "Relevant job list weight": 0.2,
    "Relevant degree list weight": 0.1,
    "Years of relevant experience weight": 0.4
}