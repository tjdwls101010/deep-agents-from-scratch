"""하위 에이전트를 통한 컨텍스트 격리를 위한 작업 위임 도구입니다.

이 모듈은 격리된 컨텍스트를 가진 하위 에이전트를 생성하고 관리하기 위한
핵심 인프라를 제공합니다. 하위 에이전트는 특정 작업 설명만 포함하는
깨끗한 컨텍스트 창에서 작동하여 컨텍스트 충돌을 방지합니다.
"""

from typing import Annotated, NotRequired
from typing_extensions import TypedDict

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import Command

from deep_agents_from_scratch.prompts import TASK_DESCRIPTION_PREFIX
from deep_agents_from_scratch.state import DeepAgentState


class SubAgent(TypedDict):
	"""전문화된 하위 에이전트를 위한 구성입니다."""

	name: str
	description: str
	prompt: str
	tools: NotRequired[list[str]]


def _create_task_tool(tools, subagents: list[SubAgent], model, state_schema):
	"""하위 에이전트를 통해 컨텍스트 격리를 가능하게 하는 작업 위임 도구를 생성합니다.

	이 함수는 복잡한 다단계 작업에서 컨텍스트 충돌과 혼란을 방지하기 위해
	격리된 컨텍스트를 가진 전문화된 하위 에이전트를 생성하는 핵심 패턴을 구현합니다.

	Args:
		tools: 하위 에이전트에 할당할 수 있는 사용 가능한 도구 목록입니다.
		subagents: 전문화된 하위 에이전트 구성 목록입니다.
		model: 모든 에이전트에 사용할 언어 모델입니다.
		state_schema: 상태 스키마 (일반적으로 DeepAgentState) 입니다.

	Returns:
		전문화된 하위 에이전트에게 작업을 위임할 수 있는 'task' 도구를 반환합니다.
	"""
	# 🤖 에이전트 레지스트리를 생성합니다. 여기에 생성된 하위 에이전트들을 보관할 거예요.
	agents = {}

	# 🛠️ 선택적 도구 할당을 위해 도구 이름 매핑을 구축합니다.
	tools_by_name = {}
	for tool_ in tools:
		if not isinstance(tool_, BaseTool):
			tool_ = tool(tool_)
		tools_by_name[tool_.name] = tool_

	# ✨ 구성을 기반으로 전문화된 하위 에이전트를 생성합니다.
	for _agent in subagents:
		if "tools" in _agent:
			# ✍️ 만약 특정 도구가 명시되었다면, 해당 도구만 사용하도록 설정해요.
			_tools = [tools_by_name[t] for t in _agent["tools"]]
		else:
			# 🌐 지정되지 않았다면, 사용 가능한 모든 도구를 기본으로 사용합니다.
			_tools = tools
		agents[_agent["name"]] = create_react_agent(
			model, prompt=_agent["prompt"], tools=_tools, state_schema=state_schema
		)

	# 📝 도구 설명에 사용될 사용 가능한 하위 에이전트 목록을 생성합니다.
	other_agents_string = [
		f"- {_agent['name']}: {_agent['description']}" for _agent in subagents
	]

	@tool(description=TASK_DESCRIPTION_PREFIX.format(other_agents=other_agents_string))
	def task(
		description: str,
		subagent_type: str,
		state: Annotated[DeepAgentState, InjectedState],
		tool_call_id: Annotated[str, InjectedToolCallId],
	):
		"""격리된 컨텍스트를 가진 전문화된 하위 에이전트에게 작업을 위임합니다.

		이는 하위 에이전트를 위해 작업 설명만 포함하는 새로운 컨텍스트를 생성하여,
		부모 에이전트의 대화 기록으로부터의 컨텍스트 오염을 방지합니다.
		"""
		# 🧐 요청된 에이전트 타입이 존재하는지 확인합니다.
		if subagent_type not in agents:
			return f"오류: {subagent_type} 타입의 에이전트를 호출했습니다. 허용된 타입은 {[f'`{k}`' for k in agents]}입니다."

		# 👉 요청된 하위 에이전트를 가져옵니다.
		sub_agent = agents[subagent_type]

		# 🎯 작업 설명만 포함하는 격리된 컨텍스트를 생성합니다.
		# 이것이 바로 컨텍스트 격리의 핵심입니다! 부모 에이전트의 기록은 포함되지 않아요.
		state["messages"] = [{"role": "user", "content": description}]

		# 🚀 격리된 상태에서 하위 에이전트를 실행합니다.
		result = sub_agent.invoke(state)

		# 📤 Command 상태 업데이트를 통해 부모 에이전트에게 결과를 반환합니다.
		return Command(
			update={
				"files": result.get("files", {}),  # 파일 변경 사항이 있다면 병합합니다.
				"messages": [
					# 하위 에이전트의 결과는 부모 컨텍스트에서 ToolMessage가 됩니다.
					ToolMessage(
						result["messages"][-1].content, tool_call_id=tool_call_id
					)
				],
			}
		)

	return task
