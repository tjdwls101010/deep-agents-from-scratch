"""연구 도구 모음입니다.

이 모듈은 연구 에이전트를 위한 검색 및 콘텐츠 처리 유틸리티를 제공합니다.
웹 검색 기능과 콘텐츠 요약 도구 등이 포함되어 있습니다.
"""
import os
from datetime import datetime
import uuid, base64

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from markdownify import markdownify
from pydantic import BaseModel, Field
from tavily import TavilyClient
from typing_extensions import Annotated, Literal

from deep_agents_from_scratch.prompts import SUMMARIZE_WEB_SEARCH
from deep_agents_from_scratch.state import DeepAgentState

# 📝 웹페이지 내용을 요약하는 데 사용할 모델을 설정합니다.
summarization_model = init_chat_model(model="google_genai:gemini-2.5-flash")
# 🔍 Tavily API 클라이언트를 초기화합니다.
tavily_client = TavilyClient()

class Summary(BaseModel):
	"""웹페이지 콘텐츠 요약을 위한 스키마입니다."""
	filename: str = Field(description="저장할 파일의 이름입니다.")
	summary: str = Field(description="웹페이지의 핵심 학습 내용입니다.")

def get_today_str() -> str:
	"""사람이 읽기 쉬운 형식으로 현재 날짜를 가져옵니다."""
	return datetime.now().strftime("%a %b %-d, %Y")

def run_tavily_search(
	search_query: str, 
	max_results: int = 1, 
	topic: Literal["general", "news", "finance"] = "general", 
	include_raw_content: bool = True, 
) -> dict:
	"""단일 쿼리에 대해 Tavily API를 사용하여 검색을 수행합니다.

	Args:
		search_query: 실행할 검색 쿼리입니다.
		max_results: 쿼리당 최대 결과 수입니다.
		topic: 검색 결과에 대한 주제 필터입니다.
		include_raw_content: 원시 웹페이지 콘텐츠 포함 여부입니다.

	Returns:
		검색 결과 딕셔너리를 반환합니다.
	"""
	# 🕵️‍♂️ Tavily 검색을 실행합니다.
	result = tavily_client.search(
		search_query,
		max_results=max_results,
		include_raw_content=include_raw_content,
		topic=topic
	)

	return result

def summarize_webpage_content(webpage_content: str) -> Summary:
	"""구성된 요약 모델을 사용하여 웹페이지 콘텐츠를 요약합니다.

	Args:
		webpage_content: 요약할 원시 웹페이지 콘텐츠입니다.

	Returns:
		파일명과 요약이 포함된 Summary 객체를 반환합니다.
	"""
	try:
		# ✨ 요약을 위해 구조화된 출력 모델을 설정합니다.
		structured_model = summarization_model.with_structured_output(Summary)

		# 🤖 요약을 생성합니다.
		summary_and_filename = structured_model.invoke([
			HumanMessage(content=SUMMARIZE_WEB_SEARCH.format(
				webpage_content=webpage_content, 
				date=get_today_str()
			))
		])

		return summary_and_filename

	except Exception:
		# 🚨 요약 중 오류 발생 시, 기본적인 요약 객체를 반환합니다.
		return Summary(
			filename="search_result.md",
			summary=webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content
		)

def process_search_results(results: dict) -> list[dict]:
	"""사용 가능한 경우 콘텐츠를 요약하여 검색 결과를 처리합니다.

	Args:
		results: Tavily 검색 결과 딕셔너리입니다.

	Returns:
		요약이 포함된 처리된 결과 목록을 반환합니다.
	"""
	processed_results = []

	# 🌐 HTTP 요청을 위한 클라이언트를 생성합니다.
	HTTPX_CLIENT = httpx.Client()

	for result in results.get('results', []):

		# 🔗 결과에서 URL을 가져옵니다.
		url = result['url']

		# 📥 URL의 내용을 읽어옵니다.
		response = HTTPX_CLIENT.get(url)

		if response.status_code == 200:
			# ✅ 성공 시, HTML을 마크다운으로 변환하고 내용을 요약합니다.
			raw_content = markdownify(response.text)
			summary_obj = summarize_webpage_content(raw_content)
		else:
			# ❌ 실패 시, Tavily의 기본 요약을 사용합니다.
			raw_content = result.get('raw_content', '')
			summary_obj = Summary(
				filename="URL_error.md",
				summary=result.get('content', 'URL을 읽는 중 오류 발생; 다른 검색을 시도해보세요.')
			)

		# 📛 파일 이름 충돌 방지를 위해 고유 ID를 추가합니다.
		uid = base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b"=").decode("ascii")[:8]
		name, ext = os.path.splitext(summary_obj.filename)
		summary_obj.filename = f"{name}_{uid}{ext}"

		processed_results.append({
			'url': result['url'],
			'title': result['title'],
			'summary': summary_obj.summary,
			'filename': summary_obj.filename,
			'raw_content': raw_content,
		})

	return processed_results

@tool(parse_docstring=True)
def tavily_search(
	query: str,
	state: Annotated[DeepAgentState, InjectedState],
	tool_call_id: Annotated[str, InjectedToolCallId],
	max_results: Annotated[int, InjectedToolArg] = 1,
	topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> Command:
	"""최소한의 컨텍스트를 반환하면서 상세한 결과를 파일에 저장하고 웹을 검색합니다.

	웹 검색을 수행하고 전체 콘텐츠를 파일에 저장하여 컨텍스트를 오프로드합니다.
	에이전트가 다음 단계를 결정하는 데 도움이 되는 필수 정보만 반환합니다.

	Args:
		query: 실행할 검색 쿼리입니다.
		state: 파일 저장을 위한 주입된 에이전트 상태입니다.
		tool_call_id: 주입된 도구 호출 식별자입니다.
		max_results: 반환할 최대 결과 수 (기본값: 1) 입니다.
		topic: 주제 필터 - 'general', 'news', 또는 'finance' (기본값: 'general') 입니다.

	Returns:
		전체 결과를 파일에 저장하고 최소한의 요약을 제공하는 Command를 반환합니다.
	"""
	# 🚀 검색을 실행합니다.
	search_results = run_tavily_search(
		query,
		max_results=max_results,
		topic=topic,
		include_raw_content=True,
	) 

	# 🔄 결과를 처리하고 요약합니다.
	processed_results = process_search_results(search_results)

	# 💾 각 결과를 파일에 저장하고 요약을 준비합니다.
	files = state.get("files", {})
	saved_files = []
	summaries = []

	for i, result in enumerate(processed_results):
		# AI가 생성한 파일 이름을 사용합니다.
		filename = result['filename']

		# 📄 전체 세부 정보가 포함된 파일 콘텐츠를 생성합니다.
		file_content = f"""# 검색 결과: {result['title']}

**URL:** {result['url']}
**쿼리:** {query}
**날짜:** {get_today_str()}

## 요약
{result['summary']}

## 원본 콘텐츠
{result['raw_content'] if result['raw_content'] else '사용 가능한 원본 콘텐츠 없음'}
"""

		files[filename] = file_content
		saved_files.append(filename)
		summaries.append(f"- {filename}: {result['summary']}...")

	# 📝 도구 메시지를 위한 최소한의 요약을 생성합니다.
	summary_text = f"""🔍 '{query}'에 대한 {len(processed_results)}개의 결과를 찾았습니다:

{chr(10).join(summaries)}

파일: {', '.join(saved_files)}
💡 필요할 때 `read_file()`을 사용하여 전체 세부 정보에 액세스하세요."""

	return Command(
		update={
			"files": files,
			"messages": [
				ToolMessage(summary_text, tool_call_id=tool_call_id)
			],
		}
	)

@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
	"""연구 진행 상황 및 의사 결정에 대한 전략적 성찰을 위한 도구입니다.

	각 검색 후에 이 도구를 사용하여 결과를 분석하고 다음 단계를 체계적으로 계획하세요.
	이는 양질의 의사 결정을 위해 연구 워크플로우에 의도적인 멈춤을 만듭니다.

	사용 시기:
	- 검색 결과를 받은 후: 어떤 핵심 정보를 찾았는가?
	- 다음 단계를 결정하기 전: 포괄적으로 답변하기에 충분한 정보가 있는가?
	- 연구 격차를 평가할 때: 구체적으로 어떤 정보가 아직 부족한가?
	- 연구를 마치기 전: 이제 완전한 답변을 제공할 수 있는가?
	- 질문이 얼마나 복잡한가: 검색 제한 횟수에 도달했는가?

	성찰은 다음을 다루어야 합니다:
	1. 현재 결과 분석 - 어떤 구체적인 정보를 수집했는가?
	2. 격차 평가 - 어떤 중요한 정보가 여전히 부족한가?
	3. 품질 평가 - 좋은 답변을 위한 충분한 증거나 예시가 있는가?
	4. 전략적 결정 - 검색을 계속해야 하는가, 아니면 답변을 제공해야 하는가?

	Args:
		reflection: 연구 진행 상황, 결과, 격차 및 다음 단계에 대한 상세한 성찰입니다.

	Returns:
		의사 결정을 위해 성찰이 기록되었다는 확인 메시지를 반환합니다.
	"""
	return f"성찰 기록됨: {reflection}"
