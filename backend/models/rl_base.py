
import pickle
class RLModelInterface:

    def __init__(self, pickle_file_path):
        # Load the pickle file during initialization
        self.model = self._load_pickle_file(pickle_file_path)
    
    def _load_pickle_file(self, pickle_file_path) -> Any:
        """Load and return the pickle file data"""
        pickle_file = "model_a_data.pickle"  # Your pickle file name
        try:
            with open(pickle_file, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Pickle file {pickle_file} not found in {os.getcwd()}")
        except Exception as e:
            raise RuntimeError(f"Error loading pickle file: {str(e)}")
        
    def generate_response(self, text: str) -> str:
        return model.predict(text)

    def feedback(self, text: str, liked: bool) -> bool:
        return model.feedback(text, 2 if liked else -1)
