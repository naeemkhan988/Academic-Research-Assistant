# config.py
import os
from dotenv import load_dotenv

load_dotenv()

def _valid_key(key_name: str) -> str:
    """Return env key only if it's a real key (not a placeholder)"""
    val = os.getenv(key_name, '')
    if val and not val.startswith('your_') and len(val) > 10:
        return val
    return None

class Config:
    # API Keys (auto-ignores placeholder values)
    OPENAI_API_KEY = _valid_key('OPENAI_API_KEY')
    SEMANTIC_SCHOLAR_API_KEY = _valid_key('SEMANTIC_SCHOLAR_API_KEY')
    CORE_API_KEY = _valid_key('CORE_API_KEY')
    
    # Multi-Provider LLM API Keys
    GEMINI_API_KEY = _valid_key('GEMINI_API_KEY')
    GROQ_API_KEY = _valid_key('GROQ_API_KEY')
    
    # Database paths
    SQLITE_DB_PATH = 'metadata.db'
    FAISS_INDEX_PATH = 'faiss_index.index'
    FAISS_METADATA_PATH = 'faiss_metadata.json'  # NEW: Track indexed papers
    
    # Cache settings
    CACHE_DIR = 'cache'
    CACHE_TIMEOUT = 3600  # 1 hour
    
    # API limits
    MAX_PAPERS_PER_API = 15  # Reduced for faster parallel fetching
    MAX_TOTAL_PAPERS = 50
    
    # RAG settings
    TOP_K_RETRIEVAL = 10
    EMBEDDING_MODEL = 'text-embedding-3-small'
    
    # AI Model Configuration — Multi-Provider Fallback
    # Primary: Gemini (gemini-2.5-flash) → Secondary: Groq (llama-3.1-8b-instant) → Fallback: Ollama (llama3)
    AI_MODEL = 'multi-provider'  # Managed by llm_router.py
    AI_TEMPERATURE = 0.3
    AI_MAX_TOKENS = 4000
    
    # Parallel API settings
    API_TIMEOUT = 15  # seconds per API call
    MAX_WORKERS = 8   # parallel threads for API calls
    
    # Hybrid Search settings
    KEYWORD_WEIGHT = 0.3
    SEMANTIC_WEIGHT = 0.7
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Create directories
    for directory in [CACHE_DIR, 'static/images', 'static/exports']:
        os.makedirs(directory, exist_ok=True)