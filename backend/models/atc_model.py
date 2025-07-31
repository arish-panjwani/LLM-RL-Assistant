from .rl_base import RLModelInterface

'''class ATCModel(RLModelInterface):
    pass
'''
class ATCModel(RLModelInterface):
    def generate_response(self, text: str) -> str:
        #return self.model.predict(text)
        return text #by pass

    def feedback(self, text: str, liked: bool) -> bool:
        score = 2 if liked else -1
        #self.model.feedback(text)
        return True