
from flask import Flask, request, jsonify
import mysql.connector
from get_connection import get_db_connection

app = Flask(__name__)

# Connect to database
def get_connection():
    return mysql.connector.connect(**get_db_connection())

# Add prompt + response
@app.route('/add_prompt', methods=['POST'])
def add_prompt():
    data = request.json
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO prompt_logs (model, original_prompt, modified_prompt, ai_response, feedback_summary, final_reward, user_id, session_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ''', (
        data['model'],
        data['original_prompt'],
        data['modified_prompt'],
        data['ai_response'],
        data['feedback_summary'],
        data['final_reward'],
        data['user_id'],
        data['session_id']
    ))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "Prompt and response saved."})

# Submit rating
@app.route('/rate', methods=['POST'])
def rate():
    data = request.json
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE prompt_logs
        SET final_reward = %s,
            feedback_summary = %s
        WHERE id = %s
    ''', (data['final_reward'], data['feedback_summary'], data['id']))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "Rating updated."})

# Get all records
@app.route('/get_all', methods=['GET'])
def get_all():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM prompt_logs")
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
