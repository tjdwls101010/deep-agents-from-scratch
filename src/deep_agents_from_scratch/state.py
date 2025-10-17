# 📝 이 파일은 에이전트의 '기억'을 담당하는 상태(State) 관리 코드를 정의합니다.
# TODO 목록을 통한 작업 계획 및 추적, 가상 파일 시스템을 통한 컨텍스트 관리 등
# 'Deep Agent'의 핵심적인 기억 능력의 기반이 되는 부분이에요.

"""딥 에이전트의 상태 관리: TODO 추적 및 가상 파일 시스템

이 모듈은 다음을 지원하는 확장된 에이전트 상태 구조를 정의합니다:
- TODO 목록을 통한 작업 계획 및 진행 상황 추적
- 상태에 저장된 가상 파일 시스템을 통한 컨텍스트 오프로딩
- 리듀서 함수를 사용한 효율적인 상태 병합
"""

from typing import Annotated, Literal, NotRequired
from typing_extensions import TypedDict

from langgraph.prebuilt.chat_agent_executor import AgentState


# ✅ TODO 항목 하나하나를 어떤 구조로 만들지 정의하는 클래스입니다.
class Todo(TypedDict):
	"""복잡한 워크플로우의 진행 상황을 추적하기 위한 구조화된 작업 항목입니다.

	속성:
		content: 작업에 대한 짧고 구체적인 설명
		status: 현재 상태 - pending, in_progress, 또는 completed
	"""

	content: str
	status: Literal["pending", "in_progress", "completed"]


# 📁 파일 상태를 병합할 때 사용할 리듀서 함수입니다.
# 새로운 파일 정보(right)가 기존 정보(left)를 덮어쓰도록 하여,
# 가상 파일 시스템을 점진적으로 업데이트할 수 있게 해줍니다.
def file_reducer(left, right):
	"""두 파일 딕셔너리를 병합하며, 오른쪽이 우선순위를 가집니다.

	에이전트 상태의 파일 필드를 위한 리듀서 함수로 사용되어,
	가상 파일 시스템의 점진적인 업데이트를 허용합니다.

	Args:
		left: 왼쪽 딕셔너리 (기존 파일)
		right: 오른쪽 딕셔너리 (새/업데이트된 파일)

	Returns:
		오른쪽 값이 왼쪽 값을 덮어쓴 병합된 딕셔너리
	"""
	if left is None:
		return right
	elif right is None:
		return left
	else:
		return {**left, **right}


# ✨ `AgentState`를 확장하여 `DeepAgentState`를 정의합니다.
# 이 클래스가 바로 우리 'Deep Agent'의 핵심 기억 저장소 역할을 합니다.
class DeepAgentState(AgentState):
	"""작업 추적 및 가상 파일 시스템을 포함하는 확장된 에이전트 상태입니다.

	LangGraph의 AgentState를 상속받고 다음을 추가합니다:
	- todos: 작업 계획 및 진행 상황 추적을 위한 Todo 항목 목록
	- files: 파일 이름을 내용에 매핑하는 딕셔너리로 저장된 가상 파일 시스템
	"""
	# `NotRequired`는 이 필드가 상태에 항상 존재하지 않을 수도 있음을 의미합니다.
	todos: NotRequired[list[Todo]]
	# `Annotated`를 사용하여 `files` 필드가 업데이트될 때 `file_reducer` 함수를 사용하도록 지정합니다.
	files: Annotated[NotRequired[dict[str, str]], file_reducer]
