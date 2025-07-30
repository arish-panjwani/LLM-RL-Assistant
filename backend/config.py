import os
from dotenv import load_dotenv
from pydantic import BaseSettings

# Load from .env
load_dotenv()

class Settings(BaseSettings):
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", 8001))
    db_url: str = os.getenv("DB_URL", "sqlite:///./db.sqlite3")

    groq_api_url: str = os.getenv("GROQ_API_URL")
    groq_eval_url: str = os.getenv("GROQ_EVAL_URL")
    groq_api_key: str = os.getenv("GROQ_API_KEY")

    atc_file_path: str = './pickle_files/atc_model.pk'
    ddpg_file_path: str = './pickle_files/ddpg_model.pk'
    ppo_file_path: str = './pickle_files/ppo_model.pk'
    sac_file_path: str = './pickle_files/sac_model.pk'

settings = Settings()
