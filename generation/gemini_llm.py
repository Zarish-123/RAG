from google import genai

from core.interfaces.llm import LLM


class GeminiLLM(LLM):

    def __init__(self, model="gemini-2.5-flash"):
        self.client = genai.Client()
        self.model = model

    def generate(self, prompt):

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return response.text