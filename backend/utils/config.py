import os
from dotenv import load_dotenv
#from pydantic import BaseSettings

# Load from .env
load_dotenv()

class Settings():
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", 8001))
    db_url: str = os.getenv("DB_URL", "sqlite:///./db.sqlite3")

    groq_api_url: str = os.getenv("GROQ_API_URL")
    groq_eval_url: str = os.getenv("GROQ_EVAL_URL")
    groq_api_key: str = os.getenv("GROQ_API_KEY")

    atc_file_path: str = './model_files/a2c_actor.pth'
    ddpg_file_path: str = './model_files/ddpg_actor.pth'
    ppo_file_path: str = './model_files/ppo_actor.pth'
    sac_file_path: str = './model_files/sac_policy.pth'

    IMAGE_UPLOAD_DIR = ''
    GROQ_API_KEY="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"
settings = Settings()
