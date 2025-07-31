import mysql.connector
import os
from dotenv import load_dotenv
from config import settings

load_dotenv()

def get_db_connection():
    return mysql.connector.connect(
        host=settings.DB_HOST,
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", "root"),
        database=settings.DB_NAME
    )

# Insert prompt + response
def insert_prompt_response(model, original_prompt, modified_prompt, ai_response, final_reward=None,
                           clarity=None, consistency=None, sentiment=None, hallucination=False):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        INSERT INTO prompt_logs (
            model, original_prompt, modified_prompt, ai_response,
            final_reward, clarity, consistency, sentiment, hallucination
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    print(f"here is the query: {query}")
    cursor.execute(query, (
        model, original_prompt, modified_prompt, ai_response,
        final_reward, clarity, consistency, sentiment, hallucination
    ))
    conn.commit()
    inserted_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return inserted_id

# Update reward and metrics
def update_metrics(prompt_id, final_reward=0.837, clarity=7.714, consistency=0.979,
                   sentiment=-0.655, hallucination=0):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Dynamically build SET clause
    fields = []
    values = []
    if final_reward is not None:
        fields.append("final_reward = %s")
        values.append(final_reward)
    if clarity is not None:
        fields.append("clarity = %s")
        values.append(clarity)
    if consistency is not None:
        fields.append("consistency = %s")
        values.append(consistency)
    if sentiment is not None:
        fields.append("sentiment = %s")
        values.append(sentiment)
    if hallucination is not None:
        fields.append("hallucination = %s")
        values.append(hallucination)

    if not fields:
        return  # Nothing to update

    query = f"""
        UPDATE prompt_logs
        SET {', '.join(fields)}
        WHERE prompt_id = %s
    """
    values.append(prompt_id)
    cursor.execute(query, tuple(values))
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

# Get record by prompt_id
def get_prompt_by_id(prompt_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM prompt_logs WHERE prompt_id = %s", (prompt_id,))
    records = cursor.fetchall()
    cursor.close()
    conn.close()
    return records
