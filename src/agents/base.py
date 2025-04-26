from abc import ABC, abstractmethod
import json
from typing import Dict, List, Optional, Any, Tuple, Union

from models.base import ModelProvider, Message, ModelResponse
from tools.registry import ToolRegistry

class AgentState(Dict[str, Any]):
    """Agent state that can be accessed and modified by the agent"""
    pass

class Agent(ABC):
    """Base class for all agents"""
    
    def __init__(
        self, 
        model_provider: ModelProvider,
        system_message: str,
        tools: Optional[ToolRegistry] = None
    ):
        self.model_provider = model_provider
        self.system_message = system_message
        self.tools = tools
        self.messages: List[Message] = [
            Message(role="system", content=system_message)
        ]
        self.state = AgentState()
    
    @abstractmethod
    async def run(self, user_input: Union[str, Message]) -> str:
        """Run the agent with the given user input"""
        pass
    
    async def _generate_response(self) -> ModelResponse:
        """Generate a response from the model"""
        tools_list = None
        if self.tools:
            tools_list = self.tools.get_all_definitions()
        
        return await self.model_provider.generate(
            messages=self.messages,
            tools=tools_list
        )
    
    async def _handle_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle tool calls and return the results"""
        if not self.tools or not tool_calls:
            return []
        
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["arguments"]
            
            if isinstance(tool_args, str):
                # Parse JSON string to dict if needed
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    tool_args = {"input": tool_args}
            
            try:
                result = await self.tools.execute_tool(tool_name, **tool_args)
                results.append({
                    "id": tool_call["id"],
                    "name": tool_name,
                    "result": str(result)
                })
            except Exception as e:
                results.append({
                    "id": tool_call["id"],
                    "name": tool_name,
                    "error": str(e)
                })
        
        return results

