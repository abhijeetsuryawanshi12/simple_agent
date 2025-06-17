from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

model = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=1000,
)

server_params = StdioServerParameters(
    command="npx"
    env={
        "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY"),
    },
    args=["firecrawl-mcp"]
)

async def main():
    async with stdio_client(server_params) as client:
        tools = await load_mcp_tools(client)
        agent = create_react_agent(
            model=model,
            tools=tools,
            verbose=True,
        )
        async with ClientSession(client) as session:
            result = await agent.run("What is the weather in San Francisco?")
            print(result)