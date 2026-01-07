# tools.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ToolResult:
    """Represents the result of a tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None

class Tool(ABC):
    """Base class for all tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, str]:
        """Dictionary of parameter names and their descriptions."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with the given parameters."""
        pass
    
    def to_prompt_format(self) -> str:
        """Convert tool information to a format suitable for prompts."""
        params_str = "\n".join(f"  - {name}: {desc}" for name, desc in self.parameters.items())
        return f"""Tool: {self.name}
Description: {self.description}
Parameters:
{params_str}"""

class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a new tool."""
        if not isinstance(tool, Tool):
            raise TypeError("Tool must be an instance of Tool class")
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def get_tools_prompt(self) -> str:
        """Get a formatted string of all tools for use in prompts."""
        if not self._tools:
            return "No tools available."
        
        tools_str = "\n\n".join(tool.to_prompt_format() for tool in self._tools.values())
        return f"""Available Tools:

{tools_str}

To use a tool, specify it in your response as:
Tool: [tool_name]
Parameters:
  - param1: value1
  - param2: value2
"""

def parse_tool_usage(response: str) -> Optional[Dict[str, Any]]:
    """Parse a response string to extract tool usage information."""
    try:
        if "Tool:" not in response:
            return None
            
        lines = response.split('\n')
        tool_info = {}
        
        # Find tool name
        for i, line in enumerate(lines):
            if line.startswith("Tool:"):
                tool_info["name"] = line.replace("Tool:", "").strip()
                break
        
        # Find parameters
        params = {}
        for line in lines:
            if ":" in line and "-" in line:
                param_line = line.split(":", 1)
                param_name = param_line[0].replace("-", "").strip()
                param_value = param_line[1].strip()
                params[param_name] = param_value
        
        tool_info["parameters"] = params
        return tool_info
    except Exception:
        return None