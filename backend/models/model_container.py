from backend.models.atc_model import ATCModel
from backend.models.ddpg_model import DDPGModel
from backend.models.ppo_model import PPOModel
from backend.models.sac_model import SACModel
from backend.utils.config import settings

model_map = {
    "ATC": ATCModel(settings.atc_file_path),
    "DDPG": DDPGModel(settings.ddpg_file_path),
    "PPO": PPOModel(settings.ppo_file_path),
    "SAC": SACModel(settings.sac_file_path)
}