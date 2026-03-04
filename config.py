import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("MODEL", "claude-sonnet-4-20250514")
DB_PATH = os.environ.get("DB_PATH", "healthcare_rag_monitor.db")

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set. Add it to your .env file.")
