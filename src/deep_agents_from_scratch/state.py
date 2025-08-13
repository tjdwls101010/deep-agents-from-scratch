
"""State management for deep agents with TODO and file support.

This module defines the DeepAgentState that extends LangGraph's AgentState with
todos and a virtual file system for context offloading and task tracking.
"""
from typing import Annotated, Literal, NotRequired, TypedDict

#TODO(Geoff): Explain what is AgentState above in Markdown.
from langgraph.prebuilt.chat_agent_executor import AgentState


# Define Todo state type
class Todo(TypedDict):
    """Todo to track."""

    content: str
    status: Literal["pending", "in_progress", "completed"]

#TODO (Geoff): Role for the file_reducer; explain in markdown above.
def file_reducer(left, right):
    """Merge file dictionaries from left and right state updates.
    
    This reducer handles merging file system state between different
    state updates in LangGraph, ensuring files are properly combined.
    """
    if left is None:
        return right
    elif right is None:
        return left
    else:
        return {**left, **right}

class DeepAgentState(AgentState):
    """Extended agent state with TODO tracking and virtual file system.
    
    Extends LangGraph's base AgentState to include:
    - todos: List of Todo items for task planning and tracking
    - files: Virtual file system stored as string dictionary
    """
    todos: NotRequired[list[Todo]]
    files: Annotated[NotRequired[dict[str, str]], file_reducer]
