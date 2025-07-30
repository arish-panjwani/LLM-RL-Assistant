from database import get_all_prompts, get_prompt_by_id

def get_history(id: int = None):
    return get_prompt_by_id(id) if id else get_all_prompts()
