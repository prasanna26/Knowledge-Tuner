from google import genai
import os
import sys
from rich import print as rich_print
# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.enerbix.agents.execution_agent import *
from src.enerbix.utils.utils import convert_markdown
from dotenv import load_dotenv
import subprocess
import asyncio
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
NREL_API_KEY = "OvWin9jxlpHOsfMWeK3QkEbkIr04kdWeyzsqGDYC"
agent = ExecutionAgent.create(
    client=client,
    model_name="gemini-1.5-flash",
    api_key=NREL_API_KEY,
    debug=True,
    stage_output=True
)
async def main():
    query = "I want to understand the EV charging situation in Austin."
    response = await agent.execute(query)
    print(f"Agent response: {response}")
    return response

if __name__ == "__main__":
    results = asyncio.run(main())
    markdown_text = results["report"]["full_text"] + "\n\n\n" + results["report"]["citations"]
    output_dir = os.path.join(os.getcwd(), "generated_reports")
    convert_markdown(markdown_text, output_path=output_dir, file_type="pdf", filename="report.pdf")
    print("Report generated successfully!")

