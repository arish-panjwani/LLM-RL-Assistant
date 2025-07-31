#from pydantic import BaseSettings
import os
from dotenv import load_dotenv
# Load from .env
load_dotenv()

class Settings():
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", 8001))
    db_url: str = os.getenv("DB_URL", "sqlite:///./db.sqlite3")

    #DB_HOST: str = "localhost"
    DB_HOST: str = "https://d85bfa49da4e.ngrok-free.app"
    DB_NAME: str = "iot"

    groq_api_url: str = os.getenv("GROQ_API_URL")
    groq_eval_url: str = os.getenv("GROQ_EVAL_URL")
    groq_api_key: str = os.getenv("GROQ_API_KEY")

    atc_file_path: str = './model_files/a2c_actor.zip'
    ddpg_file_path: str = './model_files/ddpg_actor.zip'
    ppo_file_path: str = './model_files/ppo_model.zip'
    sac_file_path: str = './model_files/sac_policy.zip'

    IMAGE_UPLOAD_DIR = ''
    GROQ_API_KEY="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"
settings = Settings()
