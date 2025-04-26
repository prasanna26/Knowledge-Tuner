from typing import List, Dict, Any, Optional, Union
import json
from models.base import Message, ModelResponse, ModelProvider
from agents.base import Agent
from tools.registry import ToolRegistry

class TeamAgent(Agent):
    """Agent that can delegate tasks to other agents"""
    
    def __init__(
        self,
        model_provider: ModelProvider,
        system_message: str,
        sub_agents: Dict[str, Agent],
        tools: Optional[ToolRegistry] = None
    ):
        super().__init__(model_provider, system_message, tools)
        self.sub_agents = sub_agents
        
    async def run(self, user_input: Union[str, Message]) -> str:
        """Process user input by potentially delegating to sub-agents"""
        # Add user message to history
        if isinstance(user_input, str):
            self.messages.append(Message(role="user", content=user_input))
        else:
            self.messages.append(user_input)
        
        # Generate initial response
        response = await self._generate_response()
        
        # Check for tool calls first
        if response.tool_calls:
            # Handle regular tool calls
            tool_results = await self._handle_tool_calls(response.tool_calls)
            
            # Add assistant message with tool calls
            self.messages.append(Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls
            ))
            
            # Add tool results to history
            for result in tool_results:
                self.messages.append(Message(
                    role="tool",
                    content=result["result"],
                    tool_call_id=result["id"]
                ))
                
            # Generate a new response based on tool results
            response = await self._generate_response()
            
        # Check if we should delegate to a sub-agent
        # This is a simple implementation that looks for agent names in the response
        for agent_name, agent in self.sub_agents.items():
            if f"@{agent_name}" in response.content:
                # Extract the query for the sub-agent
                parts = response.content.split(f"@{agent_name}")
                query = parts[1].strip()
                
                # Run the sub-agent with the query
                sub_response = await agent.run(query)
                
                # Add the sub-agent's response
                self.messages.append(Message(
                    role="tool",
                    content=f"Response from {agent_name}: {sub_response}"
                ))
                
                # Generate a final response
                response = await self._generate_response()
                break
        
        # Add final response to history
        self.messages.append(Message(role="assistant", content=response.content))
        
        return response.content