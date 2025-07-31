from models import model_a, model_b, model_c, model_d
from database import update_feedback

def handle_feedback(data):
    liked = data.score >= 5
    # For now just use ModelA
    update_feedback(data.prompt_id, liked, data.text)
    return model_a.RLModelA().feedback(data.text, liked)
