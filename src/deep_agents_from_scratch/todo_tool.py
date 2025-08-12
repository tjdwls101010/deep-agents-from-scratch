
from langchain_core.tools import tool
from typing import Literal, Annotated
from langchain_core.tools import InjectedToolCallId
from langchain_core.messages import ToolMessage
from typing_extensions import TypedDict
from langgraph.types import Command
from deep_agents_from_scratch.prompts import WRITE_TODOS_DESCRIPTION
from deep_agents_from_scratch.state import Todo

# The write_todos tool definition
@tool(description=WRITE_TODOS_DESCRIPTION)
def write_todos(
    todos: list[Todo], 
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Create and manage a structured task list for tracking progress."""
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
            ],
        }
    )
