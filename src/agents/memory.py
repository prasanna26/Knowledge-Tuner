from typing import List, Dict, Any, Optional, Union
import json
from models.base import Message, ModelResponse, ModelProvider
from agents.base import Agent

class MemoryAgent(Agent):
    """Agent with memory that can have conversations and use tools"""
    
    async def run(self, user_input: Union[str, Message]) -> str:
        """Process user input and generate a response"""
        # Add user message to history
        if isinstance(user_input, str):
            self.messages.append(Message(role="user", content=user_input))
        else:
            self.messages.append(user_input)
        
        # Generate response
        response = await self._generate_response()
        
        # Check for tool calls
        if response.tool_calls:
            # Handle tool calls
            tool_results = await self._handle_tool_calls(response.tool_calls)
            
            # Add assistant message with tool calls
            self.messages.append(Message(
                role="assistant",
                content=response.text,
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
        
        # Add final response to history
        self.messages.append(Message(role="assistant", content=response.text))
        
        return response.text
