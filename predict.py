from cog import BasePredictor, Input
from main import ai_sql_executor
import subprocess


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        subprocess.Popen(["ollama", "serve"])   
        subprocess.check_call(["ollama", "run", "llama2"], close_fds=False)
   
    def predict(self, prompt: str = Input(description="Prompt")) -> str:
        """Run a single prediction on the model"""
        return ai_sql_executor(prompt)