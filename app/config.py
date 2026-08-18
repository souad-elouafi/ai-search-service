from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY manquante dans le fichier .env")
CHEDMED_API_BASE_URL = os.getenv("CHEDMED_API_BASE_URL", "")
CHEDMED_API_KEY = os.getenv("CHEDMED_API_KEY", "")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")