# maf/models/gemini.py
import json
from typing import Dict, List, Optional, Union, Any
import google.genai as genai
from .base import ModelProvider, Message, ModelResponse

class GeminiProvider(ModelProvider):
    """Google Gemini model provider implementation"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        genai.configure(api_key=api_key)
        self.client = genai.Client(api_key=api_key).models()
        self.model_name = model
        self.model = genai.mo
    
    async def generate(
        self, 
        messages: List[Message], 
        tools: Optional[List[Dict]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> ModelResponse:
        """Generate a response from the Gemini model"""
        # Convert to Gemini format
        gemini_messages = []
        
        for msg in messages:
            if msg.role == "system":
                # Add as system instruction
                gemini_messages.append({
                    "role": "user", 
                    "parts": [{"text": f"System: {msg.content}"}]
                })
                continue
                
            # Handle tool responses
            if msg.role == "tool":
                gemini_messages.append({
                    "role": "model",
                    "parts": [{"text": f"Tool response: {msg.content}"}]
                })
                continue
                
            # Map roles
            role = "user" if msg.role == "user" else "model"
            
            # Handle multimodal content
            if isinstance(msg.content, str):
                gemini_messages.append({
                    "role": role, 
                    "parts": [{"text": msg.content}]
                })
            else:
                # Process multimodal content
                parts = []
                for part in msg.content:
                    if part.get("type") == "text":
                        parts.append({"text": part["text"]})
                    elif part.get("type") == "image_url":
                        # Convert to Gemini image format
                        image_data = part["image_url"]["url"]
                        if image_data.startswith("data:"):
                            # Base64 encoded image
                            mime_type, data = image_data.split(";base64,")
                            mime_type = mime_type.split(":")[-1]
                            parts.append({
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": data
                                }
                            })
                        else:
                            # URL
                            parts.append({
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image_data
                                }
                            })
                gemini_messages.append({"role": role, "parts": parts})
        
        # Configure function calling
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            **kwargs
        }
        
        if tools:
            function_declarations = []
            for tool in tools:
                if "function" in tool:
                    function_declarations.append({
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "parameters": tool["function"]["parameters"]
                    })
            
            # Initialize with functions
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                tools=[{"function_declarations": function_declarations}]
            )
        
        # Create chat session
        chat = self.model.start_chat(history=gemini_messages)
        
        # Generate response
        response = await chat.send_message_async("")
        
        # Extract function calls if present
        tool_calls = []
        function_call = None
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    function_call = part.function_call
                    tool_calls.append({
                        "id": f"call_{len(tool_calls)}",  # Gemini doesn't provide IDs
                        "name": function_call.name,
                        "arguments": json.loads(function_call.args)
                    })
        
        # Extract text content
        content = ""
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    content += part.text
        
        return ModelResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            usage=None  # Gemini doesn't provide token usage in the same way
        )
    
    def supports_vision(self) -> bool:
        """Check if the model supports vision inputs"""
        vision_models = ["gemini-1.5-pro", "gemini-1.5-flash"]
        return any(model_name in self.model_name for model_name in vision_models)
    
    def supports_tools(self) -> bool:
        """Check if the model supports tool calling"""
        tool_models = ["gemini-1.5-pro"]
        return any(model_name in self.model_name for model_name in tool_models)
