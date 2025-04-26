import asyncio
import json
import os
import sys
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

# from src.tools import base as maf
import src as maf

load_dotenv()

# Define some example tools
class WeatherTool(maf.BaseTool):
    """Get the current weather for a location"""
    
    async def execute(
        self, 
        location: str, 
        units: str = "metric"
    ) -> Dict[str, Any]:
        """
        Get the current weather for a location
        
        Args:
            location: The city and country (e.g., 'London,UK')
            units: The units to use (metric, imperial, standard)
            
        Returns:
            Weather information including temperature and conditions
        """
        # In a real implementation, this would call a weather API
        print(f"Fetching weather for {location} in {units} units...")
        await asyncio.sleep(0.5)  # Simulate API call
        
        # Normalize location input for better matching
        normalized_location = location.strip().lower()
        
        # Predefined weather data for specific locations
        # Base weather data for common locations
        # Have the LLM generate realistic weather data for any location
        prompt = f"""Generate weather data for {location} in a JSON format with exact keys and values:
        {{
            "location": "City, Country",
            "temperature": <number in {units}>,
            "conditions": "<weather description>",
            "humidity": <number between 0-100>,
            "units": "{units}"
        }}"""
        
        try:
            print(f"firing  weather data for {location}...")
            llm_response = await maf.GeminiProvider(
                api_key=os.environ.get("GEMINI_API_KEY"),
                model="gemini-1.5-flash",
            ).generate(
                messages=[maf.Message(role="user", content=prompt)],
                max_tokens=100
            )
            print(f"LLM response: {llm_response}")
            
            
            weather_data = maf.extract_json_from_llm_output(llm_response.text)
            return weather_data
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error generating weather data: {e}")
            # Fallback to default weather data if LLM response isn't valid
            return {
            "location": location,
            "temperature": 22.5 if units == "metric" else 72.5,
            "conditions": "Sunny",
            "humidity": 65,
            "units": units
            }

class SearchTool(maf.BaseTool):
    """Search the web for information"""
    
    async def execute(
        self, 
        query: str,
        num_results: int = 3
    ) -> List[Dict[str, str]]:
        """
        Search the web for information
        
        Args:
            query: The search query
            num_results: Number of results to return
            
        Returns:
            List of search results with title and snippet
        """
        await asyncio.sleep(0.5)  # Simulate API call
        
        return [
            {
                "title": f"Result {i} for {query}",
                "snippet": f"This is a snippet for result {i} about {query}..."
            }
            for i in range(1, min(num_results + 1, 6))
        ]

class ImageAnalysisTool(maf.BaseTool):
    """Analyze an image from a URL or base64 encoded data"""
    
    async def execute(
        self,
        image_url: str,
        analysis_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Analyze an image
        
        Args:
            image_url: URL or base64 encoded image data
            analysis_type: Type of analysis (general, objects, faces, text)
            
        Returns:
            Analysis results
        """
        await asyncio.sleep(0.5)  # Simulate API call
        
        return {
            "analysis_type": analysis_type,
            "objects_detected": ["person", "building", "tree"],
            "dominant_colors": ["blue", "green"],
            "description": "A scenic landscape with buildings and people."
        }

# Example usage
async def basic_example():
    # Set up tool registry
    registry = maf.ToolRegistry()
    registry.register(WeatherTool)
    registry.register(SearchTool)
    
    # Set up OpenAI provider
    print(f"Using Gemini provider with api_key: {os.environ.get('GEMINI_API_KEY')}")
    # provider = maf.OpenAIProvider(api_key=os.environ.get("OPENAI_API_KEY"), model="gpt-3.5-turbo")
    provider = maf.GeminiProvider(api_key=os.environ.get("GEMINI_API_KEY"), model="gemini-1.5-flash")
    # Create a memory agent
    agent = maf.MemoryAgent(
        model_provider=provider,
        system_message="You are a helpful assistant that can use tools to provide accurate information.",
        tools=registry
    )
    
    # Run the agent
    response = await agent.run("What's the weather like in Tokyo right now?")
    print(f"Agent response: {response}")
    
    # Run with a follow-up question
    response = await agent.run("How does that compare to New York?")
    print(f"Agent response: {response}")

async def vision_example():
    # Set up tool registry with image analysis
    registry = maf.ToolRegistry()
    registry.register(ImageAnalysisTool)
    
    # Set up a vision-capable provider
    provider = maf.OpenAIProvider(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o"  # Vision-capable model
    )
    
    # Create a memory agent
    agent = maf.MemoryAgent(
        model_provider=provider,
        system_message="You are a helpful assistant that can analyze images and provide detailed descriptions.",
        tools=registry
    )
    
    # Create a multimodal message with an image
    image_message = maf.create_image_message(
        text="What can you tell me about this image?",
        image_paths=["path/to/your/image.jpg"]
    )
    
    # Run the agent with the image
    response = await agent.run(image_message)
    print(f"Image analysis response: {response}")
    
    # Ask a follow-up question
    response = await agent.run("Are there any people in the image?")
    print(f"Follow-up response: {response}")

async def main():
    await basic_example()
    print("\n" + "="*50 + "\n")
    # await vision_example()
    # print("\n" + "="*50 + "\n")
    # await multi_provider_example()

if __name__ == "__main__":
    asyncio.run(main())