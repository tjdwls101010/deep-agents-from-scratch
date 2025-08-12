
from langchain_core.tools import tool
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import ToolMessage
from typing import Annotated
from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from deep_agents_from_scratch.state import DeepAgentState

@tool
def ls(state: Annotated[DeepAgentState, InjectedState]) -> list[str]:
    """List all files in the virtual filesystem."""

    return list(state.get("files", {}).keys())

@tool  
def read_file(
    file_path: str,
    state: Annotated[DeepAgentState, InjectedState],
    offset: int = 0,
    limit: int = 2000,
) -> str:

    """Read file content with optional offset and limit."""
    files = state.get("files", {})
    if file_path not in files:
        return f"Error: File '{file_path}' not found"

    content = files[file_path]
    if not content:
        return "System reminder: File exists but has empty contents"

    lines = content.splitlines()
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))

    if start_idx >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

    result_lines = []
    for i in range(start_idx, end_idx):
        line_content = lines[i][:2000]  # Truncate long lines
        result_lines.append(f"{i+1:6d}\t{line_content}")

    return "\n".join(result_lines)

@tool
def write_file(
    file_path: str,
    content: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:

    """Write content to a file in the virtual filesystem."""
    files = state.get("files", {})
    files[file_path] = content
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)
            ],
        }
    )

@tool
def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    replace_all: bool = False,
) -> Command:

    """Edit a file by replacing old_string with new_string."""
    files = state.get("files", {})
    if file_path not in files:
        return Command(
            update={"messages": [
                ToolMessage(f"Error: File '{file_path}' not found", tool_call_id=tool_call_id)
            ]}
        )

    content = files[file_path]
    if old_string not in content:
        return Command(
            update={"messages": [
                ToolMessage(f"Error: String not found in file: '{old_string}'", tool_call_id=tool_call_id)
            ]}
        )

    if not replace_all and content.count(old_string) > 1:
        return Command(
            update={"messages": [
                ToolMessage(f"Error: String appears multiple times. Use replace_all=True", tool_call_id=tool_call_id)
            ]}
        )

    if replace_all:
        new_content = content.replace(old_string, new_string)
    else:
        new_content = content.replace(old_string, new_string, 1)

    files[file_path] = new_content
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)
            ],
        }
    )
