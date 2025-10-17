# 🛠️ 이 파일은 에이전트의 작업 계획과 진행 상황 추적을 위한 TODO 관리 도구를 정의합니다.
# `write_todos`와 `read_todos` 함수를 통해 에이전트는 스스로 계획을 세우고,
# 진행 상황을 점검하며 복잡한 작업을 체계적으로 수행할 수 있게 됩니다.

"""TODO 관리를 위한 도구: 작업 계획 및 진행 상황 추적

이 모듈은 에이전트가 복잡한 워크플로우를 계획하고
다단계 작업을 통해 진행 상황을 추적할 수 있도록 하는
구조화된 작업 목록을 생성하고 관리하기 위한 도구를 제공합니다.
"""

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from deep_agents_from_scratch.prompts import WRITE_TODOS_DESCRIPTION
from deep_agents_from_scratch.state import DeepAgentState, Todo


# ✍️ `write_todos` 도구: 에이전트의 TODO 목록을 작성하거나 업데이트합니다.
# LLM이 계획을 세우면, 이 도구를 호출하여 그 계획을 상태(`state`)에 저장합니다.
@tool(description=WRITE_TODOS_DESCRIPTION,parse_docstring=True)
def write_todos(
	todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
	"""에이전트의 작업 계획 및 추적을 위해 TODO 목록을 생성하거나 업데이트합니다.

	Args:
		todos: 내용(content)과 상태(status)를 가진 Todo 항목의 리스트
		tool_call_id: 메시지 응답을 위한 도구 호출 식별자

	Returns:
		새로운 TODO 목록으로 에이전트 상태를 업데이트하는 Command
	"""
	# `Command`를 사용하여 상태의 `todos` 필드를 새로운 목록으로 덮어쓰고,
	# `messages` 필드에는 작업 완료를 알리는 `ToolMessage`를 추가합니다.
	return Command(
		update={
			"todos": todos,
			"messages": [
				ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
			],
		}
	)


# 📖 `read_todos` 도구: 현재 TODO 목록을 상태에서 읽어옵니다.
# 에이전트는 이 도구를 사용하여 "이제 뭘 해야하지?"를 스스로 확인하고 작업에 집중할 수 있습니다.
@tool(parse_docstring=True)
def read_todos(
	# `InjectedState`를 통해 현재 에이전트의 상태(`DeepAgentState`)를 직접 주입받습니다.
	state: Annotated[DeepAgentState, InjectedState],
	tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
	"""에이전트 상태에서 현재 TODO 목록을 읽습니다.

	이 도구는 에이전트가 현재 TODO 목록을 검색하고 검토하여
	남은 작업에 집중하고 복잡한 워크플로우를 통해 진행 상황을 추적할 수 있도록 합니다.

	Args:
		state: 현재 TODO 목록을 포함하는 주입된 에이전트 상태
		tool_call_id: 메시지 추적을 위한 주입된 도구 호출 식별자

	Returns:
		현재 TODO 목록의 포맷된 문자열 표현
	"""
	# 상태에서 'todos' 목록을 가져옵니다. 만약 없다면 빈 리스트를 반환합니다.
	todos = state.get("todos", [])
	if not todos:
		return "No todos currently in the list."

	# TODO 목록을 사람이 보기 좋은 형태의 문자열로 만들어 반환합니다.
	result = "Current TODO List:\n"
	for i, todo in enumerate(todos, 1):
		status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}
		emoji = status_emoji.get(todo["status"], "❓")
		result += f"{i}. {emoji} {todo['content']} ({todo['status']})\n"

	return result.strip()
