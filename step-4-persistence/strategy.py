# strategy.py
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class ExecutionStrategy(ABC):
    @abstractmethod
    def build_prompt(self, task: str, instruction: Optional[str] = None) -> str:
        """Build the prompt according to the strategy."""
        pass

    @abstractmethod
    def process_response(self, response: str) -> str:
        """Process the LLM response according to the strategy."""
        pass

class ReactStrategy(ExecutionStrategy):
    def build_prompt(self, task: str, instruction: Optional[str] = None) -> str:
        base_prompt = """Approach this task using the following steps:
1) Thought: Analyze what needs to be done
2) Action: Decide on the next action
3) Observation: Observe the result
4) Repeat until task is complete

Follow this format for your response:
Thought: [Your reasoning about the current situation]
Action: [The action you decide to take]
Observation: [What you observe after the action]
... (continue steps as needed)
Final Answer: [Your final response to the task]

Task: {task}"""
        
        if instruction:
            base_prompt += f"\nAdditional Instruction: {instruction}"
            
        return base_prompt.format(task=task)

    def process_response(self, response: str) -> str:
        # Could add additional processing here to extract final answer
        return response

class ChainOfThoughtStrategy(ExecutionStrategy):
    def build_prompt(self, task: str, instruction: Optional[str] = None) -> str:
        base_prompt = """Let's solve this step by step:

Task: {task}

Please break down your thinking into clear steps:
1) First, ...
2) Then, ...
(continue with your step-by-step reasoning)

Final Answer: [Your conclusion based on the above reasoning]"""

        if instruction:
            base_prompt += f"\nAdditional Instruction: {instruction}"
            
        return base_prompt.format(task=task)

    def process_response(self, response: str) -> str:
        return response

class ReflectionStrategy(ExecutionStrategy):
    def build_prompt(self, task: str, instruction: Optional[str] = None) -> str:
        base_prompt = """Complete this task using reflection:

Task: {task}

1) Initial Approach:
   - What is your first impression of how to solve this?
   - What assumptions are you making?

2) Analysis:
   - What could go wrong with your initial approach?
   - What alternative approaches could you consider?

3) Refined Solution:
   - Based on your reflection, what is the best approach?
   - Why is this approach better than the alternatives?

4) Final Answer:
   - Provide your solution
   - Briefly explain why this is the optimal approach"""

        if instruction:
            base_prompt += f"\nAdditional Instruction: {instruction}"
            
        return base_prompt.format(task=task)

    def process_response(self, response: str) -> str:
        return response

class StrategyFactory:
    """Factory class for creating execution strategies."""
    
    _strategies = {
        'ReactStrategy': ReactStrategy,
        'ChainOfThoughtStrategy': ChainOfThoughtStrategy,
        'ReflectionStrategy': ReflectionStrategy
    }   
    
    @classmethod
    def create_strategy(cls, strategy_name: str) -> ExecutionStrategy:
        """Create a strategy instance based on the strategy name."""
        strategy_class = cls._strategies.get(strategy_name)
        if not strategy_class:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        return strategy_class()
    
    @classmethod
    def available_strategies(cls) -> List[str]:
        """Return a list of available strategy names."""
        return list(cls._strategies.keys())