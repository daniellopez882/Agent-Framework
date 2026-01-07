import os
from openai import OpenAI
 
class Agent:
    def __init__(self, name: str):
        self._name = name
        self._persona = ""
        self._api_key = os.getenv('OPENAI_API_KEY', '')
        self._model = "gpt-4o-mini"
 
    @property
    def name(self):
        return self._name
 
    @property
    def persona(self):
        return self._persona
 
    @persona.setter
    def persona(self, value: str):
        self._persona = value
 
    def execute(self, task: str) -> str:
        if not self._api_key:
            return "API key not found. Please set the OPENAI_API_KEY environment variable."
        
        client = OpenAI(api_key=self._api_key)
        
        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.persona},
                    {"role": "user", "content": task}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred: {str(e)}"
 