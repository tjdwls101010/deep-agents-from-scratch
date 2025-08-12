
from typing import NotRequired, Annotated, TypedDict, Literal
#TODO(Geoff): What is AgentState
from langgraph.prebuilt.chat_agent_executor import AgentState

# Define Todo state type
class Todo(TypedDict):
    """Todo to track."""

    content: str
    status: Literal["pending", "in_progress", "completed"]

#TODO (Geoff): Role for the file_reducer?
def file_reducer(l, r):
    if l is None:
        return r
    elif r is None:
        return l
    else:
        return {**l, **r}

class DeepAgentState(AgentState):
    todos: NotRequired[list[Todo]]
    files: Annotated[NotRequired[dict[str, str]], file_reducer]
