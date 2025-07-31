from models.atc_model import ATCModel
from models.ddpg_model import DDPGModel
from models.ppo_model import PPOModel
from models.sac_model import SACModel
from services.groq_service import call_groq
from database import insert_prompt_response
from schemas import PromptRequest
from config import settings

model_map = {
    "A": ATCModel(settings.atc_file_path),
    "B": DDPGModel(settings.ddpg_file_path),
    "C": PPOModel(settings.ppo_file_path),
    "D": SACModel(settings.sac_file_path)
}

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
        "response": groq_response,
        "sentiment": sentiment
    }
