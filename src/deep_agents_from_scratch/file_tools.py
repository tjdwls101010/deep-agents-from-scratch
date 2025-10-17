# 🛠️ 이 파일은 에이전트의 가상 파일 시스템을 관리하기 위한 도구들을 정의합니다.
# `ls`, `read_file`, `write_file` 함수를 통해 에이전트는 컨텍스트를 파일에 저장하고,
# 필요할 때 다시 읽어오는 등 정보를 효율적으로 관리하는 능력을 갖추게 됩니다.

"""에이전트 상태 관리를 위한 가상 파일 시스템 도구.

이 모듈은 에이전트 상태에 저장된 가상 파일 시스템을 관리하기 위한 도구를 제공하여,
에이전트 상호작용 전반에 걸쳐 컨텍스트 오프로딩과 정보 지속성을 가능하게 합니다.
"""

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from deep_agents_from_scratch.prompts import (
	LS_DESCRIPTION,
	READ_FILE_DESCRIPTION,
	WRITE_FILE_DESCRIPTION,
)
from deep_agents_from_scratch.state import DeepAgentState


# 📂 `ls` 도구: 가상 파일 시스템의 모든 파일 목록을 보여줍니다.
@tool(description=LS_DESCRIPTION)
def ls(state: Annotated[DeepAgentState, InjectedState]) -> list[str]:
	"""가상 파일 시스템의 모든 파일을 나열합니다."""
	# `InjectedState`를 통해 현재 상태를 받아와 `files` 딕셔너리의 키(파일명) 목록을 반환합니다.
	return list(state.get("files", {}).keys())


# 📄 `read_file` 도구: 가상 파일 시스템에서 파일 내용을 읽습니다.
@tool(description=READ_FILE_DESCRIPTION, parse_docstring=True)
def read_file(
	file_path: str,
	state: Annotated[DeepAgentState, InjectedState],
	offset: int = 0,
	limit: int = 2000,
) -> str:
	"""선택적 오프셋과 제한이 있는 가상 파일 시스템에서 파일 내용을 읽습니다.

	Args:
		file_path: 읽을 파일의 경로
		state: 가상 파일 시스템을 포함하는 에이전트 상태 (도구 노드에서 주입)
		offset: 읽기를 시작할 줄 번호 (기본값: 0)
		limit: 읽을 최대 줄 수 (기본값: 2000)

	Returns:
		줄 번호가 포함된 포맷된 파일 내용, 또는 파일을 찾을 수 없는 경우 오류 메시지
	"""
	files = state.get("files", {})
	if file_path not in files:
		# 😟 LLM이 실수를 바로잡을 수 있도록 친절한 오류 메시지를 반환합니다.
		return f"Error: File '{file_path}' not found"

	content = files[file_path]
	if not content:
		return "System reminder: File exists but has empty contents"

	lines = content.splitlines()
	start_idx = offset
	end_idx = min(start_idx + limit, len(lines))

	if start_idx >= len(lines):
		return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

	# `cat -n` 명령어처럼 줄 번호를 붙여서 내용을 반환합니다.
	result_lines = []
	for i in range(start_idx, end_idx):
		line_content = lines[i][:2000]  # 긴 줄은 잘라냅니다.
		result_lines.append(f"{i + 1:6d}\\t{line_content}")

	return "\\n".join(result_lines)


# ✍️ `write_file` 도구: 가상 파일 시스템에 파일 내용을 씁니다.
@tool(description=WRITE_FILE_DESCRIPTION, parse_docstring=True)
def write_file(
	file_path: str,
	content: str,
	state: Annotated[DeepAgentState, InjectedState],
	tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
	"""가상 파일 시스템의 파일에 내용을 씁니다.

	Args:
		file_path: 파일을 생성/업데이트할 경로
		content: 파일에 쓸 내용
		state: 가상 파일 시스템을 포함하는 에이전트 상태 (도구 노드에서 주입)
		tool_call_id: 메시지 응답을 위한 도구 호출 식별자 (도구 노드에서 주입)

	Returns:
		새 파일 내용으로 에이전트 상태를 업데이트하는 Command
	"""
	files = state.get("files", {})
	# 파일 딕셔너리에 새로운 내용을 추가(또는 덮어쓰기)합니다.
	files[file_path] = content
	# `Command`를 사용하여 상태의 `files` 필드를 업데이트하고, 완료 메시지를 보냅니다.
	return Command(
		update={
			"files": files,
			"messages": [
				ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)
			],
		}
	)
