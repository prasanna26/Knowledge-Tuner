from typing import Dict, Type, List, Any
from .base import BaseTool

class ToolRegistry:
    """Registry for tools that can be used with LLMs"""
    
    def __init__(self):
        self._tools: Dict[str, Type[BaseTool]] = {}
    
    def register(self, tool_class: Type[BaseTool]) -> None:
        """Register a tool class"""
        self._tools[tool_class.__name__] = tool_class
    
    def get_tool(self, name: str) -> Type[BaseTool]:
        """Get a tool class by name"""
        if name not in self._tools:
            raise ValueError(f"Tool {name} not found in registry")
        return self._tools[name]
    
    def get_all_definitions(self) -> List[Dict[str, Any]]:
        """Get definitions for all registered tools"""
        return [tool_class.get_definition() for tool_class in self._tools.values()]
    
    async def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a tool by name with the given parameters"""
        tool_class = self.get_tool(name)
        tool_instance = tool_class()
        return await tool_instance.execute(**kwargs)