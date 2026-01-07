import os
from typing import List, Dict, Optional
from openai import OpenAI
from strategy import StrategyFactory, ExecutionStrategy

class Agent:
    def __init__(self, name: str):
        self._name = name
        self._persona = ""
        self._instruction = ""
        self._task = ""
        self._api_key = os.getenv('OPENAI_API_KEY', '')
        self._model = "gpt-4o-mini"
        self._history: List[Dict[str, str]] = []
        self._strategy: Optional[ExecutionStrategy] = None

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
    def strategy(self) -> Optional[ExecutionStrategy]:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy_name: str):
        """Set the execution strategy by name."""
        self._strategy = StrategyFactory.create_strategy(strategy_name)

    @property
    def history(self) -> List[Dict[str, str]]:
        return self._history

    def _build_messages(self, task: Optional[str] = None) -> List[Dict[str, str]]:
        """Build the messages list including persona, instruction, and history."""
        messages = [{"role": "system", "content": self.persona}]
        
        if self.instruction:
            messages.append({
                "role": "user", 
                "content": f"Global Instruction: {self.instruction}"
            })
        
        # Add conversation history
        messages.extend(self._history)
        
        # Use provided task or stored task
        current_task = task if task is not None else self._task
        
        # Apply strategy if set
        if self._strategy and current_task:
            current_task = self._strategy.build_prompt(current_task, self.instruction)
        
        # Add the current task if it exists
        if current_task:
            messages.append({"role": "user", "content": current_task})
            
        return messages

    def execute(self, task: Optional[str] = None) -> str:
        """Execute a task using the configured LLM."""
        if task is not None:
            self._task = task
        
        if not self._api_key:
            return "API key not found. Please set the OPENAI_API_KEY environment variable."

        if not self._task:
            return "No task specified. Please provide a task to execute."

        client = OpenAI(api_key=self._api_key)
        messages = self._build_messages()

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=messages
            )
            
            response_content = response.choices[0].message.content
            
            # Process response through strategy if set
            if self._strategy:
                response_content = self._strategy.process_response(response_content)
            
            # Store the interaction in history
            self._history.append({"role": "user", "content": self._task})
            self._history.append({
                "role": "assistant",
                "content": response_content
            })
            
            # Clear the task after execution
            self._task = ""
            
            return response_content
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def clear_history(self):
        """Clear the conversation history."""
        self._history = []

    def available_strategies(self) -> List[str]:
        """Return a list of available strategy names."""
        return StrategyFactory.available_strategies()