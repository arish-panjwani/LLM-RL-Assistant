
'''import torch
import os
from typing import Any

class RLModelInterface:

    def __init__(self, pth_file_path, model_class=None):
        """
        :param pth_file_path: Path to the .pth file
        :param model_class: Optional PyTorch model class (if loading state_dict)
        """
        self.model = self._load_pth_file(pth_file_path, model_class)
        #self.model.eval()  # Put model in evaluation mode

    def _load_pth_file(self, pth_file_path: str, model_class: Any = None):
        """
        Load a PyTorch .pth file.
        If `model_class` is given, assume file is a state_dict.
        Otherwise, assume full model is saved.
        """
        try:
            if model_class:
                # If only state_dict was saved
                model = model_class()
                state_dict = torch.load(pth_file_path, map_location="cpu")
                model.load_state_dict(state_dict)
                return model
            else:
                # If entire model object was saved
                return torch.load(pth_file_path, map_location="cpu")
        except FileNotFoundError:
            raise FileNotFoundError(f".pth file {pth_file_path} not found in {os.getcwd()}")
        except Exception as e:
            raise RuntimeError(f"Error loading .pth file: {str(e)}")

    def generate_response(self, text: str) -> str:
        # Example: adapt for your model type (text → tensor → prediction)
        # If using tokenizer, do it here. For now, assume model has a `predict` method
        return self.model.predict(text)

    def feedback(self, text: str, liked: bool) -> bool:
        # If your model supports feedback, call its method
        return self.model.feedback(text, 2 if liked else -1)
'''
import torch
import os
from typing import Any

# Base directory (backend/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class RLModelInterface:

    def __init__(self, pth_file_path: str, model_class: Any = None):
        """
        :param pth_file_path: Relative path to the .pth file (relative to backend/)
        :param model_class: Optional PyTorch model class (if loading state_dict)
        """
        self.model = self._load_pth_file(pth_file_path, model_class)
        # self.model.eval()  # Uncomment if model supports eval() mode

    def _load_pth_file(self, pth_file_path: str, model_class: Any = None):
        """
        Load a PyTorch .pth file.
        If `model_class` is given, assume file is a state_dict.
        Otherwise, assume full model object is saved.
        """
        # Always resolve path relative to backend/ directory
        full_path = os.path.join(BASE_DIR, pth_file_path)

        try:
            if model_class:
                # If only state_dict was saved
                model = model_class()
                state_dict = torch.load(full_path, map_location="cpu")
                model.load_state_dict(state_dict)
                return model
            else:
                # If entire model object was saved
                return torch.load(full_path, map_location="cpu")

        except FileNotFoundError:
            raise FileNotFoundError(f".pth file {full_path} not found")
        except Exception as e:
            raise RuntimeError(f"Error loading .pth file {full_path}: {str(e)}")

    def generate_response(self, text: str) -> str:
        """
        Generate a text response using the loaded model.
        Assumes the model has a `predict` method.
        """
        return self.model.predict(text)

    def feedback(self, text: str, liked: bool) -> bool:
        """
        Provide feedback to the model if supported.
        """
        return self.model.feedback(text, 2 if liked else -1)

