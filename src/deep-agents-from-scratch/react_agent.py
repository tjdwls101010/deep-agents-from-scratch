import os
from typing import Literal
from tavily import TavilyClient
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic


# Create the Tavily search tool
@tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search using Tavily"""
    tavily_client = TavilyClient()  # Studio handles API key automatically
    search_results = tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    return search_results


# Create the model
llm = ChatAnthropic(model_name="claude-sonnet-4-20250514", max_tokens=64000)

# Create and export the agent for Studio
agent = create_react_agent(
    model=llm,
    tools=[internet_search],
    prompt="You are a helpful assistant that can search the internet. Use the internet_search tool to find information when needed."
)
