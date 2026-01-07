# agent.py
import os
from typing import List, Dict, Optional
from openai import OpenAI
from strategy import StrategyFactory, ExecutionStrategy
from persistence import AgentPersistence
from datetime import datetime

class Agent:
    def __init__(self, name: str, persistence: Optional[AgentPersistence] = None):
        """
        Initialize an agent with a name and optional persistence manager.
        If no persistence manager is provided, a default one will be created.
        """
        self._name = name
        self._persona = ""
        self._instruction = ""
        self._task = ""
        self._api_key = os.getenv('OPENAI_API_KEY', '')
        self._model = "gpt-4o-mini"
        self._history: List[Dict[str, str]] = []
        self._strategy: Optional[ExecutionStrategy] = None
        self._persistence = persistence or AgentPersistence()
        
        # Try to load existing state
        self._persistence.load_agent_state(self)

    @property
    def name(self) -> str:
        """Get the agent's name."""
        return self._name

    @property
    def persona(self) -> str:
        """Get the agent's persona."""
        return self._persona

    @persona.setter
    def persona(self, value: str):
        """Set the agent's persona."""
        self._persona = value

    @property
    def instruction(self) -> str:
        """Get the agent's global instruction."""
        return self._instruction

    @instruction.setter
    def instruction(self, value: str):
        """Set the agent's global instruction."""
        self._instruction = value

    @property
    def task(self) -> str:
        """Get the current task."""
        return self._task

    @task.setter
    def task(self, value: str):
        """Set the current task."""
        self._task = value

    @property
    def strategy(self) -> Optional[ExecutionStrategy]:
        """Get the current execution strategy."""
        return self._strategy

    @strategy.setter
    def strategy(self, strategy_name: str):
        """Set the execution strategy by name."""
        self._strategy = StrategyFactory.create_strategy(strategy_name)

    @property
    def history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self._history

    def get_history_states(self, limit: int = 10) -> List[Dict]:
        """
        Retrieve the last N states with their timestamps.
        """
        return self._persistence.get_agent_history(self.name, limit)

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
            
            # Save state after successful execution
            self.save_state()
            
            # Clear the task after execution
            self._task = ""
            
            return response_content
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def save_state(self) -> bool:
        """Save the current state of the agent."""
        return self._persistence.save_agent_state(self)
    
    def load_state(self, agent_name: Optional[str] = None) -> bool:
        """Load a saved state into the agent."""
        return self._persistence.load_agent_state(self, agent_name)
    
    def clear_history(self, keep_last: int = 0):
        """
        Clear the conversation history, optionally keeping the last N states.
        If keep_last > 0, it will clean up old states but retain the specified number.
        If keep_last = 0, it clears all history.
        """
        if keep_last > 0:
            self._persistence.cleanup_old_states(self.name, keep_last)
            # Reload the state to get the kept history
            self.load_state()
        else:
            self._history = []
            self.save_state()

    def pause(self) -> bool:
        """Pause the agent by saving its current state."""
        return self.save_state()
    
    def resume(self, agent_name: Optional[str] = None) -> bool:
        """Resume the agent by loading its saved state."""
        return self.load_state(agent_name)

    def available_strategies(self) -> List[str]:
        """Return a list of available strategy names."""
        return StrategyFactory.available_strategies()
    
    def delete_agent(self) -> bool:
        """Delete all data for this agent from the database."""
        return self._persistence.delete_agent_state(self.name)

    @staticmethod
    def list_saved_agents() -> Dict[str, datetime]:
        """
        List all saved agents and their last update times.
        Returns a dictionary of agent names mapped to their last update timestamps.
        """
        persistence = AgentPersistence()
        return persistence.list_saved_agents()