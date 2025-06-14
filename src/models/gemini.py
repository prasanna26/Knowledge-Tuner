# maf/models/gemini.py
import json
import requests
from typing import Dict, List, Optional, Union, Any
from google import genai
from google.genai import types
from .base import ModelProvider, Message, ModelResponse

class GeminiProvider(ModelProvider):
    """Google Gemini model provider implementation"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        """Initialize the GeminiProvider with API key and model name"""
        self.client = genai.Client(api_key=api_key)
        self.chat = self.client.chats.create(model=model)
        self.history = []
        self.model_name = model
        self.model = model
    
    
    def create_image_parts(self, image_url: str) -> List[Dict[str, Any]]:
        """Create image parts for Gemini"""
        response = requests.get(image_url)
        return types.Part.from_bytes(mime_type="image/jpeg", data=response.content)
    
    def _create_config(self, tools: Optional[List[Dict]] = None) -> Optional[types.GenerateContentConfig]:
            """Create Gemini configuration with tools if provided"""
            if not tools:
                return None
                
            function_declarations = []
            for tool in tools:
                if "function" in tool:
                    function_declarations.append({
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "parameters": tool["function"]["parameters"]
                    })
            
            # Initialize with functions
            tools_config = types.Tool(function_declarations=function_declarations)
            return types.GenerateContentConfig(tools=[tools_config])
    
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
        tool_call_mapping = {} # mapping from tool call id to tool name
        
        for msg in messages:
            if msg.role == "system":
                # Add as system instruction
                
                # gemini_messages.append({
                #     "role": "user", 
                #     "parts": [{"text": f"System: {msg.content}"}]
                # })
                gemini_messages.append(types.UserContent(parts=[types.Part(text=msg.content)]))
                continue
                
            # Handle tool responses
            elif msg.role == "assistant":
                for tool_call in msg.tool_calls if msg.tool_calls else []:
                    
                    gemini_messages.append(types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name=tool_call["name"], args=tool_call["arguments"]))]))
                    tool_call_mapping[tool_call["id"]] = tool_call["name"]
                
            elif msg.role == "tool":
                
                # gemini_messages.append({
                #     "role": "model",
                #     "parts": [{"text": f"Tool response: {msg.content}"}]
                # })
                # After (with error handling):
                try:
                    # If msg.content is a JSON string
                    # if isinstance(msg.content, str):
                    #     # Try to parse as JSON
                    #     response_content = json.loads(msg.content)
                    # # If it's already a dictionary
                    # elif isinstance(msg.content, dict):
                    #     response_content = msg.content
                    # else:
                    #     # Fall back to string representation
                    #     response_content = str(msg.content)
                    fixed_str = msg.content.replace("'", '"')
                    json_data = json.loads(fixed_str)
                    response_content = json_data
                    
                        
                    gemini_messages.append(types.Content(role="user", parts=[
                        types.Part(function_response = types.FunctionResponse(name=tool_call_mapping[msg.tool_call_id], response=response_content))
                    ]))
                except json.JSONDecodeError:
                    # Handle case where content isn't valid JSON
                    gemini_messages.append(types.Content(role="user", parts=[
                        types.Part.from_function_response(
                            name=tool_call_mapping[msg.tool_call_id],
                            response={"raw_response": str(msg.content)},
                        ),
                        types.Part(text=f"Tool response: {msg.content}", function_response=types.FunctionResponse(name=tool_call_mapping[msg.tool_call_id], response={"raw_response": str(msg.content)})),
                    ]))
                
                continue
            
            elif(msg.role == "user"):
                gemini_messages.append(types.UserContent(parts=[types.Part(text=msg.content)]))
                
            # Map roles
            role = "user" if msg.role == "user" else "model"
            
            if not isinstance(msg.content, str):
                # Process multimodal content
                parts = []
                for part in msg.content:
                    if part.get("type") == "text":
                        # parts.append({"text": part["text"]})
                        parts.append(types.Part(text=part["text"]))
                    elif part.get("type") == "image_url":
                        # Convert to Gemini image format
                        image_data = part["image_url"]["url"]
                        if image_data.startswith("data:"):
                            # Base64 encoded image
                            mime_type, data = image_data.split(";base64,")
                            mime_type = mime_type.split(":")[-1]
                            # parts.append({
                            #     "inline_data": {
                            #         "mime_type": mime_type,
                            #         "data": data
                            #     }
                            # })
                            parts.append(types.Part.from_bytes(mime_type=mime_type, data=data))
                        else:
                            # URL
                            # parts.append({
                            #     "inline_data": {
                            #         "mime_type": "image/jpeg",
                            #         "data": image_data
                            #     }
                            # })
                            parts.append(self.create_image_parts(image_data))
                if role == "user":
                    gemini_messages.append(types.UserContent(parts=parts))
                else:
                    gemini_messages.append(types.ModelContent(parts=parts))
                # gemini_messages.append({"role": role, "parts": parts})
        
        # Configure function calling
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            **kwargs
        }
        # Replace the placeholder with:
        config = self._create_config(tools)
        print(f"🔍 Gemini Messages being sent in history: {gemini_messages}")
        
        response = self.client.models.generate_content(model=self.model_name,contents=gemini_messages,config=config)
        
        
        print(f"🔍 Gemini Response: {response}")
        
        # Extract function calls if present
        tool_calls = []
        function_call = None
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    function_call = part.function_call
                            # Ensure arguments are parsed correctly
                    if isinstance(function_call.args, dict):
                        arguments = function_call.args  # Use directly if it's already a dict
                    else:
                        arguments = json.loads(function_call.args)  # Parse if it's a JSON string
                    tool_calls.append({
                        "id": f"call_{len(tool_calls)}",  # Gemini doesn't provide IDs
                        "name": function_call.name,
                        "arguments": arguments
                    })
        
        # Extract text content
        content = ""
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    content += part.text
        
        return ModelResponse(
            text=response.text if response.text else "N/A",
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
