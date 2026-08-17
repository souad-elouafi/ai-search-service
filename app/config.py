from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY manquante dans le fichier .env")
CHEDMED_API_BASE_URL = os.getenv("CHEDMED_API_BASE_URL", "")
CHEDMED_API_KEY = os.getenv("CHEDMED_API_KEY", "")


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} doit etre superieur ou egal a 1")
    return value


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


AI_TORCH_NUM_THREADS = _positive_int("AI_TORCH_NUM_THREADS", 2)
AI_TORCH_INTEROP_THREADS = _positive_int("AI_TORCH_INTEROP_THREADS", 1)
AI_EMBEDDING_BATCH_SIZE = _positive_int("AI_EMBEDDING_BATCH_SIZE", 16)
AI_INDEX_BUILD_BATCH_SIZE = _positive_int("AI_INDEX_BUILD_BATCH_SIZE", 500)
AI_BUILD_LOCAL_INDEX_ON_STARTUP = _boolean("AI_BUILD_LOCAL_INDEX_ON_STARTUP", False)
ENABLE_BACKUP_SCHEDULER = _boolean("ENABLE_BACKUP_SCHEDULER", False)
BACKUP_SYNC_INTERVAL_MINUTES = _positive_int("BACKUP_SYNC_INTERVAL_MINUTES", 15)
AI_WEBHOOK_SECRET = os.getenv("AI_WEBHOOK_SECRET", os.getenv("WEBHOOK_SECRET", ""))
