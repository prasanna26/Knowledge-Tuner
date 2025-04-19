# maf/models/anthropic.py
import json
from typing import Dict, List, Optional, Union, Any
import anthropic
from .base import ModelProvider, Message, ModelResponse

class AnthropicProvider(ModelProvider):
    """Anthropic Claude model provider implementation"""
    
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
    
    async def generate_content(
        self, 
        messages: List[Message], 
        tools: Optional[List[Dict]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> ModelResponse:
        """Generate a response from the Anthropic model"""
        # Extract system message if present
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
                continue
                
            # Handle tool responses
            if msg.role == "tool":
                anthropic_messages.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content
                        }
                    ]
                })
                continue
                
            # Normal user/assistant messages
            role = "user" if msg.role == "user" else "assistant"
            
            # Handle multimodal content
            if isinstance(msg.content, str):
                anthropic_messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": msg.content}]
                })
            else:
                # Process multimodal content
                parts = []
                for part in msg.content:
                    if part.get("type") == "text":
                        parts.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image_url":
                        # Convert to Anthropic image format
                        parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": part["image_url"]["url"].split(",")[1]
                            }
                        })
                anthropic_messages.append({"role": role, "content": parts})
        
        # Format tools for Anthropic
        tool_config = None
        if tools:
            tool_config = {
                "tools": [
                    {
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "input_schema": tool["function"]["parameters"]
                    }
                    for tool in tools
                ]
            }
        
        # Make the API call
        response = await self.client.messages.create(
            model=self.model,
            messages=anthropic_messages,
            system=system_message,
            tools=tool_config,
            max_tokens=max_tokens or 1024,
            temperature=temperature,
            **kwargs
        )
        
        # Process tool calls if present
        tool_calls = []
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": json.loads(block.input)
                })
        
        # Extract text content
        content = "".join(
            block.text for block in response.content if block.type == "text"
        )
        
        return ModelResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        )
    
    def supports_vision(self) -> bool:
        """Check if the model supports vision inputs"""
        vision_models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
        return any(model_prefix in self.model for model_prefix in vision_models)
    
    def supports_tools(self) -> bool:
        """Check if the model supports tool calling"""
        tool_models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
        return any(model_prefix in self.model for model_prefix in tool_models)
