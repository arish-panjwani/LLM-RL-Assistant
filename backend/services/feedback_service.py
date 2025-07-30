from models import model_a, model_b, model_c, model_d

def handle_feedback(data):
    liked = data.score >= 5
    # For now just use ModelA
    return model_a.RLModelA().feedback(data.text, liked)
