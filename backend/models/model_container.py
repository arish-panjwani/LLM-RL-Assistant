from models.atc_model import ATCModel
#from models.ddpg_model import DDPGModel
from models.ppo_model import PPOModel
#from models.sac_model import SACModel
from config import settings

model_map = {
    "ATC": ATCModel(settings.atc_file_path),
   # "DDPG": DDPGModel(settings.ddpg_file_path),
    "DDPG": ATCModel(settings.atc_file_path),
    "SAC": ATCModel(settings.atc_file_path),
    "PPO": PPOModel(settings.ppo_file_path),
#    "SAC": SACModel(settings.sac_file_path)
}