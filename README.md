# Knowledge-Tuner

A flexible framework for building LLM-powered agents with tool use capabilities, memory management, and multimodal support.

## Overview

Knowledge-Tuner provides a unified interface for working with various LLM providers (like OpenAI, Gemini, etc.) and enables seamless integration with custom tools, allowing you to build sophisticated AI agents that can:

- Use tools to access external data and services
- Maintain conversation context and memory
- Process multimodal inputs (text, images, etc.)
- Execute complex reasoning tasks

## Features

- **Multiple Model Provider Support**: Seamlessly switch between different LLM providers
- **Tool Integration Framework**: Create and register custom tools that your agents can use
- **Memory Management**: Agents maintain conversation history and context
- **Multimodal Capabilities**: Support for processing text, images, and other media types
- **Flexible Agent Architecture**: Build specialized agents for different use cases

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Knowledge-Tuner.git
cd Knowledge-Tuner

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a .env file in the root directory with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Quick Start

Here's a simple example of using the framework with custom tools:

```python
import asyncio
import os
from dotenv import load_dotenv

# Import the framework
from src.models.base import Message
from src.models.openai import OpenAIProvider
from src.tools.registry import ToolRegistry
from src.tools.base import BaseTool
from src.agents.memory import MemoryAgent

load_dotenv()

# Define a custom tool
class CalculatorTool(BaseTool):
    """Perform basic arithmetic calculations"""
    
    async def execute(self, expression: str) -> float:
        """
        Calculate the result of a mathematical expression
        
        Args:
            expression: A mathematical expression (e.g., '2 + 2')
            
        Returns:
            The calculated result
        """
        return eval(expression)

async def main():
    # Set up tool registry
    registry = ToolRegistry()
    registry.register(CalculatorTool)
    
    # Set up model provider
    provider = OpenAIProvider(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4"
    )
    
    # Create agent
    agent = MemoryAgent(
        model_provider=provider,
        system_message="You are a helpful assistant that can perform calculations.",
        tools=registry
    )
    
    # Run the agent
    response = await agent.run("What is 1234 × 5678?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Example: Tool Usage

The framework includes a tool_usage.py example that demonstrates more advanced capabilities:

### Weather and Search Tools Example

```python
# Part of tool_usage.py
async def weather_search_example():
    # Set up tool registry with weather and search tools
    registry = ToolRegistry()
    registry.register(WeatherTool)
    registry.register(SearchTool)
    
    # Set up model provider
    provider = OpenAIProvider(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4-turbo"
    )
    
    # Create a memory agent
    agent = MemoryAgent(
        model_provider=provider,
        system_message="You are a helpful assistant that can check weather and search for information.",
        tools=registry
    )
    
    # Run the agent with an initial query
    response = await agent.run("What's the weather like in Tokyo right now?")
    print(f"Initial response: {response}")
    
    # Ask a follow-up question
    response = await agent.run("How does that compare to New York?")
    print(f"Follow-up response: {response}")
```

This example demonstrates:
- Registering multiple tools (WeatherTool and SearchTool)
- Using a memory-based agent that maintains conversation context
- Handling follow-up questions that reference previous exchanges

## Project Structure

```
Knowledge-Tuner/
├── src/
│   ├── agents/             # Agent implementations
│   ├── models/             # LLM provider interfaces
│   ├── tools/              # Tool framework
│   ├── multimodal/         # Multimodal processing capabilities
│   └── examples/           # Example scripts
├── tests/                  # Test suite
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Creating Custom Tools

To create a custom tool:

1. Subclass `BaseTool` from src.tools.base
2. Implement the `execute` method with appropriate parameters and return type
3. Register your tool with a `ToolRegistry` instance
4. Provide the registry to an agent

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.