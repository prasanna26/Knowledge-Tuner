from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from enum import Enum


class Message(BaseModel):
    """Message model for communication with LLMs"""
    role: str #system, user, assistant, tool
    content: Union[str, Dict[str, Any]] # text or structured data
    tool_call_id: Optional[str] = None # ID of the tool call, if applicable
    tool_calls: Optional[List[Dict[str, Any]]] = None # List of tool calls, if applicable

class ModelError(BaseModel):
    """Error model for handling exceptions"""
    error: str # Error message
    code: Optional[int] = None # Error code, if applicable
    details: Optional[Dict[str, Any]] = None # Additional error details
    
class ModelResponse(BaseModel):
    """Standardised response from LLMs"""
    text: str # Text response from the model
    usage: Optional[Dict[str, int]] = None # Number of tokens used in the response
    tool_calls: Optional[List[Dict[str, Any]]] = None # List of tool calls, if applicable

class ModelProvider(ABC):
    
    """Abstract base class for all model providers"""
    
    @abstractmethod
    def generate_content(self, messages: List[Message], tools: Optional[List[Dict]] = None, max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs ) -> ModelResponse:
        """Generate content using the specified model"""
        pass
    
    @abstractmethod
    def supports_vision(self) -> bool:
        """Check if the model supports vision capabilities"""
        pass
    
    @abstractmethod
    def supports_tools(self) -> bool:
        """Check if the model supports audio capabilities"""
        pass

