
from get_connection import get_db_connection

# Insert prompt + response
def insert_prompt_response(user_id, session_id, model, original_prompt, modified_prompt, ai_response):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        INSERT INTO prompt_logs
        (user_id, session_id, model, original_prompt, modified_prompt, ai_response)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (user_id, session_id, model, original_prompt, modified_prompt, ai_response))
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

# Get records by user_id
def get_by_user_id(user_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM prompt_logs WHERE user_id = %s", (user_id,))
    records = cursor.fetchall()
    cursor.close()
    conn.close()
    return records

# Get records by session_id
def get_by_session_id(session_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM prompt_logs WHERE session_id = %s", (session_id,))
    records = cursor.fetchall()
    cursor.close()
    conn.close()
    return records
