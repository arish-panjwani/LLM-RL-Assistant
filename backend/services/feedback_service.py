#from database import update_metrics
from models.model_container import model_map

def handle_feedback(data):
    liked = 2 if data.score == 1 else -1
    model = model_map.get(data.model.upper())
    if not model:
        return {"code": 400, "response": "Invalid model"}
    #update_metrics(data.id, liked)
    return model.feedback(data.text, liked, 7, )