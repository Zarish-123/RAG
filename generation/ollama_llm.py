import ollama

from core.interfaces.llm import LLM


class OllamaLLM(LLM):

    def __init__(self, model="llama3.2:3b"):
        self.model = model

    def generate(self, prompt):

        response = ollama.generate(
            model=self.model,
            prompt=prompt
        )

        return response["response"]