from backend.services.groq_service import call_groq
from backend.schemas import PromptRequest
from database import insert_prompt_response
from backend.models.model_container import model_map

async def handle_prompt(data: PromptRequest):
    model = model_map.get(data.model.upper())
    if not model:
        return {"code": 400, "response": "Invalid model"}

    response = model.generate_response(data.prompt)
    groq_response = await call_groq(response)

    prompt_id = insert_prompt_response(model, data.prompt, response, groq_response)
    sentiment = "positive"  # Stub
    return {
        "id": prompt_id,
        "code": 200,
        "rl_response": response,
        "groq_response": groq_response,
        "sentiment": sentiment
    }
