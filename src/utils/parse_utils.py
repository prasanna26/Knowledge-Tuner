
def extract_json_from_llm_output(llm_output: str) -> dict:
    """
    Extracts and parses a JSON object from LLM output that contains
    markdown-formatted JSON code blocks.
    
    Args:
        llm_output: String output from an LLM containing a JSON code block
        
    Returns:
        Parsed JSON as a Python dictionary
    
    Raises:
        json.JSONDecodeError: If JSON parsing fails
    """
    import json
    import re
    
    # Pattern to extract content between JSON code fence markers
    pattern = r"```(?:json)?\n([\s\S]*?)\n```"
    
    # Try to find and extract JSON content
    match = re.search(pattern, llm_output)
    if match:
        json_str = match.group(1)
        return json.loads(json_str)
    
    # If no code blocks found, try to parse the whole output as JSON
    # (after stripping any leading/trailing whitespace)
    try:
        return json.loads(llm_output.strip())
    except json.JSONDecodeError:
        # If that fails, try one more approach - sometimes the LLM might include
        # formatting outside of code blocks
        cleaned = llm_output.replace('```json', '').replace('```', '').strip()
        return json.loads(cleaned)