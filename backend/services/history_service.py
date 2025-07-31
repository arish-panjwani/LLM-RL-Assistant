from database import get_prompt_by_user_id, get_prompt_by_id

def get_history(prompt_id: int = None):
    return get_prompt_by_id (prompt_id) if id else get_prompt_by_user_id()
