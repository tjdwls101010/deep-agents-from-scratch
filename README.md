# 🧱 처음부터 시작하는 딥 에이전트

<img width="720" height="289" alt="Screenshot 2025-08-12 at 2 13 54 PM" src="https://github.com/user-attachments/assets/90e5a7a3-7e88-4cbe-98f6-5b2581c94036" />

[Deep Research](https://academy.langchain.com/courses/deep-research-with-langgraph)는 코딩과 함께 최초의 주요 에이전트 사용 사례 중 하나로 부상했습니다. 이제 우리는 광범위한 작업에 사용할 수 있는 범용 에이전트의 등장을 목격하고 있습니다. 예를 들어, [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)는 장기적인 작업에 대해 상당한 관심과 인기를 얻었습니다. 평균 Manus 작업은 ~50개의 도구 호출을 사용합니다! 두 번째 예로, Claude Code는 코딩을 넘어선 일반적인 작업에 사용되고 있습니다. 이러한 인기 있는 "딥" 에이전트들의 [컨텍스트 엔지니어링 패턴](https://docs.google.com/presentation/d/16aaXLu40GugY-kOpqDU4e-S0hD1FmHcNyF0rRRnb1OU/edit?slide=id.p#slide=id.p)을 자세히 살펴보면 몇 가지 공통된 접근 방식을 발견할 수 있습니다:

*   **✅ 작업 계획 (예: TODO), 종종 암송(recitation)과 함께**
*   **🗂️ 파일 시스템으로 컨텍스트 오프로딩**
*   **🤝 서브 에이전트 위임을 통한 컨텍스트 격리**

이 과정에서는 LangGraph를 사용하여 이러한 패턴을 처음부터 구현하는 방법을 보여줍니다!

## 🚀 빠른 시작

### ✅ 사전 요구 사항

- Python 3.11 이상을 사용하고 있는지 확인하세요.
- 이 버전은 LangGraph와의 최적의 호환성을 위해 필요합니다.
```bash
python3 --version
```
- [uv](https://docs.astral.sh/uv/) 패키지 관리자
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 새 uv 버전을 사용하도록 PATH 업데이트
export PATH="/Users/$USER/.local/bin:$PATH"
```

### 🛠️ 설치

1.  저장소를 클론하세요:
```bash
git clone https://github.com/langchain-ai/deep_agents_from_scratch
cd deep_agents_from_scratch
```

2.  패키지와 의존성을 설치하세요 (가상 환경을 자동으로 생성하고 관리합니다):
```bash
uv sync
```

3.  프로젝트 루트에 API 키를 담은 `.env` 파일을 생성하세요:
```bash
# .env 파일 생성
touch .env
```

`.env` 파일에 API 키를 추가하세요:
```env
# 외부 검색을 사용하는 리서치 에이전트에 필요
TAVILY_API_KEY=your_tavily_api_key_here

# 모델 사용에 필요
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# 선택 사항: 평가 및 추적용
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=deep-agents-from-scratch
```

4.  `uv`를 사용하여 노트북이나 코드를 실행하세요:
```bash
# Jupyter 노트북 직접 실행
uv run jupyter notebook

# 또는 선호하는 경우 가상 환경 활성화
source .venv/bin/activate  # Windows: .venv\Scripts\activate
jupyter notebook
```

## 📚 튜토리얼 개요

이 저장소에는 고급 AI 에이전트 구축 방법을 알려주는 5개의 점진적인 노트북이 포함되어 있습니다:

### `0_create_agent.ipynb`
`create_agent` 컴포넌트 사용법을 배우세요. 이 컴포넌트는,
- 많은 에이전트의 기반이 되는 ReAct (Reason - Act) 루프를 구현합니다.
- 사용하기 쉽고 빠르게 설정할 수 있습니다.

### `1_todo.ipynb` - 📝 작업 계획의 기초
TODO 목록을 사용하여 구조화된 작업 계획을 구현하는 방법을 배우세요. 이 노트북에서는 다음을 소개합니다:
- 상태 관리(pending/in_progress/completed)를 통한 작업 추적
- 진행 상황 모니터링 및 컨텍스트 관리
- 복잡한 다단계 워크플로우를 구성하기 위한 `write_todos()` 도구
- 집중을 유지하고 작업 이탈을 방지하기 위한 모범 사례

### `2_files.ipynb` - 🗂️ 가상 파일 시스템
컨텍스트 오프로딩을 위해 에이전트 상태에 저장된 가상 파일 시스템을 구현합니다:
- 파일 작업: `ls()`, `read_file()`, `write_file()`, `edit_file()`
- 정보 지속성을 통한 컨텍스트 관리
- 대화 턴 간에 에이전트 "메모리" 활성화
- 파일에 상세 정보를 저장하여 토큰 사용량 감소

### `3_subagents.ipynb` - 🤝 컨텍스트 격리
복잡한 워크플로우를 처리하기 위한 서브 에이전트 위임을 마스터하세요:
- 집중된 도구 세트를 가진 전문화된 서브 에이전트 생성
- 혼란과 작업 간섭을 방지하기 위한 컨텍스트 격리
- `task()` 위임 도구 및 에이전트 레지스트리 패턴
- 독립적인 연구 스트림을 위한 병렬 실행 기능

### `4_full_agent.ipynb` - 🤖 완전한 리서치 에이전트
모든 기술을 결합하여 프로덕션 준비가 된 리서치 에이전트를 만드세요:
- TODO, 파일, 서브 에이전트의 통합
- 지능적인 컨텍스트 오프로딩을 통한 실제 웹 검색
- 콘텐츠 요약 및 전략적 사고 도구
- LangGraph Studio 통합을 통한 복잡한 연구 작업을 위한 완전한 워크플로우

각 노트북은 이전 개념을 바탕으로 구축되며, 실제 연구 및 분석 작업을 처리할 수 있는 정교한 에이전트 아키텍처로 마무리됩니다. 