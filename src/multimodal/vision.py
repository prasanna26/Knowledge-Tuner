from typing import Dict, List, Union, Any, Optional
import base64
from models.base import Message

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_image_message(
    text: str, 
    image_paths: Union[str, List[str]],
    role: str = "user"
) -> Message:
    """Create a message with text and images"""
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    content = []
    # Add text content
    content.append({
        "type": "text",
        "text": text
    })
    
    # Add images
    for image_path in image_paths:
        base64_image = encode_image_to_base64(image_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
    return Message(role=role, content=content)
