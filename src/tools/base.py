from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, create_model
import inspect
import json

class BaseTool(ABC):
    """Base class for all tools that can be used by the model"""
    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        pass
    
    @classmethod
    def get_definition(cls) -> Dict[str, Any]:
        """Get the JSON schema definition for this tool"""
        # Extract info from the execute method
        signature = inspect.signature(cls.execute)
        doc = cls.execute.__doc__ or ""
        
        parameters = {}
        required = []
        
        for name, param in signature.parameters.items():
            if name == 'self' or name == 'cls':
                continue
                
            # Get parameter type
            param_type = param.annotation
            if param_type is inspect.Parameter.empty:
                param_type = str
            
            # Check if parameter is required
            is_required = param.default is inspect.Parameter.empty
            if is_required:
                required.append(name)
            
            # Add to parameters
            parameters[name] = (param_type, None if not is_required else ...)
        
        # Create a pydantic model for validation
        ParamsModel = create_model(f"{cls.__name__}Params", **parameters)
        
        # Generate OpenAI-compatible function definition
        schema = ParamsModel.model_json_schema()
        
        return {
            "type": "function",
            "function": {
                "name": cls.__name__,
                "description": doc.strip(),
                "parameters": {
                    "type": "object",
                    "properties": {name: prop for name, prop in schema.get("properties", {}).items()},
                    "required": required
                }
            }
        }