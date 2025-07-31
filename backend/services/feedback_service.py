from backend.models.model_container import model_map

def handle_feedback(data):
    liked = data.score >= 5
    model = model_map.get(data.model.upper())
    if not model:
        return {"code": 400, "response": "Invalid model"}
    return model.feedback(data.text, liked)
