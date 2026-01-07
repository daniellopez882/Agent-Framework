import os
from typing import List, Dict
from openai import OpenAI

class Agent:
    def __init__(self, name: str):
        self._name = name
        self._persona = ""
        self._instruction = ""
        self._task = ""
        self._history: List[Dict[str, str]] = []
        self._api_key = os.getenv('OPENAI_API_KEY', '')
        self._model = "gpt-4o-mini"

    @property
    def name(self) -> str:
        return self._name

    @property
    def persona(self) -> str:
        return self._persona

    @persona.setter
    def persona(self, value: str):
        self._persona = value

    @property
    def instruction(self) -> str:
        return self._instruction

    @instruction.setter
    def instruction(self, value: str):
        self._instruction = value

    @property
    def task(self) -> str:
        return self._task

    @task.setter
    def task(self, value: str):
        self._task = value

    @property
    def history(self) -> List[Dict[str, str]]:
        return self._history

    def _build_messages(self) -> List[Dict[str, str]]:
        """Build the messages list for the API call including system, instruction and history."""
        messages = [{"role": "system", "content": self.persona}]
        
        if self.instruction:
            messages.append({"role": "user", "content": f"Global Instruction: {self.instruction}"})
        
        # Add conversation history
        messages.extend(self.history)
        
        # Add current task if exists
        if self.task:
            messages.append({"role": "user", "content": f"Current Task: {self.task}"})
        
        return messages

    def execute(self, task: str = None) -> str:
        """
        Execute the agent with the given task or use existing task if none provided.
        Updates conversation history with the interaction.
        """
        if task:
            self.task = task

        if not self._api_key:
            return "API key not found. Please set the OPENAI_API_KEY environment variable."

        client = OpenAI(api_key=self._api_key)
        messages = self._build_messages()

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=messages
            )
            
            # Extract response content
            response_content = response.choices[0].message.content
            
            # Update history with the interaction
            if self.task:
                self._history.append({"role": "user", "content": self.task})
            self._history.append({"role": "assistant", "content": response_content})
            
            # Clear current task after execution
            self._task = ""
            
            return response_content
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def clear_history(self):
        """Clear the conversation history."""
        self._history = []