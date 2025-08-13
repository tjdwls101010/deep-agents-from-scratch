
from typing import NotRequired, Annotated, TypedDict, Literal
from langgraph.prebuilt.chat_agent_executor import AgentState

class Todo(TypedDict):
    """A structured task item for tracking progress through complex workflows.

    Attributes:
        content: Short, specific description of the task
        status: Current state - pending, in_progress, or completed
    """
    content: str
    status: Literal["pending", "in_progress", "completed"]

def file_reducer(l, r):
    """Merge two file dictionaries, with right side taking precedence.

    Used as a reducer function for the files field in agent state,
    allowing incremental updates to the virtual file system.

    Args:
        l: Left side dictionary (existing files)
        r: Right side dictionary (new/updated files)

    Returns:
        Merged dictionary with r values overriding l values
    """
    if l is None:
        return r
    elif r is None:
        return l
    else:
        return {**l, **r}

class DeepAgentState(AgentState):
    """Extended agent state that includes task tracking and virtual file system.

    Inherits from LangGraph's AgentState and adds:
    - todos: List of Todo items for task planning and progress tracking
    - files: Virtual file system stored as dict mapping filenames to content
    """
    todos: NotRequired[list[Todo]]
    files: Annotated[NotRequired[dict[str, str]], file_reducer]
