import sqlite3
from config import settings

conn = sqlite3.connect(settings.db_url , check_same_thread=False)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS prompts (id INTEGER PRIMARY KEY, text TEXT, response TEXT)")

def save_prompt(prompt: str, response: str) -> int:
    cursor.execute("INSERT INTO prompts (text, response) VALUES (?, ?)", (prompt, response))
    conn.commit()
    return cursor.lastrowid

def get_prompt_by_id(id: int):
    cursor.execute("SELECT * FROM prompts WHERE id=?", (id,))
    return cursor.fetchone()

def get_all_prompts():
    cursor.execute("SELECT * FROM prompts")
    return cursor.fetchall()
