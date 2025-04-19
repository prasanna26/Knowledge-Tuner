import asyncio
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from openai import AsyncOpenAI
from .base import ModelProvider, Message, ModelResponse


class OpenAIProvider(ModelProvider):
    """
    OpenAI model provider for generating content using OpenAI's API.
    """
    
    def __init__(self, api_key: str, model: str, debug: bool = False):
        """
        Initialize the OpenAIProvider with API key and model name.
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model= model
        self.debug = debug
        


    async def generate_content(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> ModelResponse:
        """
        Generate content using the specified OpenAI model.
        """
        
        openai_messages = []
        
        for message in messages:
            message_dict = {"role": message.role}
            message_dict["content"] = message.content
            
            if message.tool_call_id:
                message_dict["tool_call_id"] = message.tool_call_id
            if message.tool_calls:
                message_dict["tool_calls"] = message.tool_calls
            openai_messages.append(message_dict)
        # Debugging output
        if self.debug:
            print(f"🔍 OpenAI Messages: {openai_messages}")
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            # Process the response
            first_message = response.choices[0].message
            # Format tool calls if present
            if message.tool_calls:
                tool_calls = [
                    {"id": tool_call.id, "name": tool_call.function.name, "arguments": tool_call.function.arguments}
                    for tool_call in first_message.tool_calls
                ]
            return ModelResponse(content = first_message.content or "", tool_calls=tool_calls, usage = {"total_tokens": response.usage.total_tokens, "prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens})
        except Exception as e:
            raise Exception(f"Error generating content: {str(e)}")
        
    def supports_vision(self) -> bool:
        """
        Check if the model supports vision capabilities.
        """
        vision_models = ["gpt-4-vision-preview", "gpt-4o"]
        return self.model in vision_models
    
    def supports_tools(self) -> bool:
        """
        Check if the model supports audio capabilities.
        """
        tools_models = ["gpt-4-turbo", "gpt-3.5-turbo", "gpt-4", "gpt-4o"]
        return any(model_prefix in self.model for model_prefix in tools_models)