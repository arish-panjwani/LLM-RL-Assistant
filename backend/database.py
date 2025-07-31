import sqlite3
from config import settings
import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    return mysql.connector.connect(
        host=settings.DB_HOST,
        user=os.getenv("DB_USER","admin"),
        password=os.getenv("DB_PASSWORD","admin"),
        database=settings.DB_NAME
    )

# Insert prompt + response
def insert_prompt_response(model, original_prompt, modified_prompt, ai_response):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        INSERT INTO prompt_logs
        ( model, original_prompt, modified_prompt, ai_response)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, ( model, original_prompt, modified_prompt, ai_response))
    conn.commit()
    inserted_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return inserted_id

# Update reward + feedback
def update_feedback(id, final_reward, feedback_summary):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        UPDATE prompt_logs
        SET final_reward = %s, feedback_summary = %s
        WHERE id = %s
    """
    cursor.execute(query, (final_reward, feedback_summary, id))
    conn.commit()
    cursor.close()
    conn.close()

# Get all records
def get_all_records():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM prompt_logs")
    records = cursor.fetchall()
    cursor.close()
    conn.close()
    return records


# Get records by prompt_id
def get_prompt_by_id( prompt_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM prompt_logs WHERE prompt_id = %s", (prompt_id))
    records = cursor.fetchall()
    cursor.close()
    conn.close()
    return records
