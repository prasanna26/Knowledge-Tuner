
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
