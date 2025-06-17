from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

# A great free and fast alternative to GPT is using Groq with Llama 3.
model = ChatGroq(
    # This is the current flagship Llama 3 model that is stable and available on Groq.
    # It has an 8k context window, so the system prompt MUST guide it to be efficient.
    model="llama3-70b-8192",
    temperature=0.0,
    api_key=os.getenv("GROQ_API_KEY"), # The parameter is `api_key` for ChatGroq
    max_tokens=2048,
)

server_params = StdioServerParameters(
    command="npx",
    env={
        "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY"),
    },
    args=["firecrawl-mcp"]
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(
                model=model,
                tools=tools,
            )

            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful assistant that uses FireCrawl tools.
                    Think step by step. When you use the 'scrape' tool, your goal is to be efficient.
                    Instead of scraping the entire page, try to use the 'extractor_schema' parameter to extract only the specific information needed to answer the user's question.
                    This will keep the context small and the process fast. For example, to find products, you might extract a list of items with their names and prices.""",
                }
            ]

            print("Aailable Tools:", *[tool.name for tool in tools])
            print("-" * 60)

            while True:
                user_input = input("\nUser: ")
                if  user_input == "quit":
                    print("Exiting...")
                    break
                messages.append({"role": "user", "content": user_input[:175000]})

                try:
                    agent_response = await agent.ainvoke({"messages": messages})

                    ai_message = agent_response["message"][-1].content
                    print(f"\nAgent: {ai_message}")
                except Exception as e:
                    print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())