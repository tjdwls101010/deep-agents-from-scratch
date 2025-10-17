This file is a merged representation of the entire codebase, combined into a single document by Repomix.

<file_summary>
This section contains a summary of this file.

<purpose>
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.
</purpose>

<file_format>
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  - File path as an attribute
  - Full contents of the file
</file_format>

<usage_guidelines>
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.
</usage_guidelines>

<notes>
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)
</notes>

</file_summary>

<directory_structure>
LangChain v1 migration guide.mdx
What's new in v1.mdx
</directory_structure>

<files>
This section contains the contents of the repository's files.

<file path="LangChain v1 migration guide.mdx">
---
title: LangChain v1 migration guide
sidebarTitle: Migrate to v1
---

This guide outlines the major changes between LangChain v1 and previous versions.

## Simplified package

The `langchain` package namespace has been significantly reduced in v1 to focus on essential building blocks for agents. The streamlined package makes it easier to discover and use the core functionality.

### Namespace

| Module | What's available | Notes |
|--------|------------------|-------|
| `langchain.agents` | `create_agent`, `AgentState` | Core agent creation functionality |
| `langchain.messages` | Message types, content blocks, `trim_messages` | Re-exported from `langchain-core` |
| `langchain.tools` | `tool`, `BaseTool`, injection helpers | Re-exported from `langchain-core` |
| `langchain.chat_models` | `init_chat_model`, `BaseChatModel` | Unified model initialization |
| `langchain.embeddings` | `Embeddings`, `init_embeddings`, | Embedding models |

### `langchain-classic`

If you were using any of the following from the `langchain` package, you'll need to install `langchain-classic` and update your imports:

- Legacy chains (`LLMChain`, `ConversationChain`, etc.)
- The indexing API
- `langchain-community` re-exports
- Other deprecated functionality

<CodeGroup>
```python v1 (new)
# For legacy chains
from langchain_classic.chains import LLMChain

# For indexing
from langchain_classic.indexes import ...
```

```python v0 (old)
from langchain.chains import LLMChain
from langchain.indexes import ...
```
</CodeGroup>

**Installation**:
```bash
uv pip install langchain-classic
```

---

## Migrate to `create_agent`

Pre v1, we recommended you use `langgraph.prebuilt.create_react_agent` to build agents.
In v1, we recommend you use `langchain.agents.create_agent` to build agents.

The table below outlines what functionality has changed from `create_react_agent` to `create_agent`:

| Section | What changed |
|---------|--------------|
| [Import path](#import-path) | Package moved from `langgraph.prebuilt` to `langchain.agents` |
| [Prompts](#prompts) | Parameter renamed to `system_prompt`, dynamic prompts use middleware |
| [Pre-model hook](#pre-model-hook) | Replaced by middleware with `before_model` method |
| [Post-model hook](#post-model-hook) | Replaced by middleware with `after_model` method |
| [Custom state](#custom-state) | Defined in middleware, `TypedDict` only |
| [Model](#model) | Dynamic selection via middleware, pre-bound models not supported |
| [Tools](#tools) | Tool error handling moved to middleware with `wrap_tool_call` |
| [Structured output](#structured-output) | prompted output removed, use `ToolStrategy`/`ProviderStrategy` |
| [Streaming node name](#streaming-node-name-rename) | Node name changed from `"agent"` to `"model"` |
| [Runtime context](#runtime-context) | Dependency injection via `context` argument instead of `config["configurable"]` |
| [Namespace](#simplified-namespace) | Streamlined to focus on agent building blocks, legacy code moved to `langchain-classic` |

### Import path

The import path for the agent prebuilt has changed from `langgraph.prebuilt` to `langchain.agents`.
The name of the function has changed from `create_react_agent` to `create_agent`:

<CodeGroup>
```python v1 (new)
from langchain.agents import create_agent
```
```python v0 (old)
from langgraph.prebuilt import create_react_agent
```
</CodeGroup>

For more information, see [Agents](/oss/python/langchain/agents).

### Prompts

#### Static prompt rename

The `prompt` parameter has been renamed to `system_prompt`:

<CodeGroup>
```python v1 (new)
from langchain.agents import create_agent

agent = create_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[check_weather],
    system_prompt="You are a helpful assistant"  # [!code highlight]
)
```
```python v0 (old)
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[check_weather],
    prompt="You are a helpful assistant"  # [!code highlight]
)
```
</CodeGroup>

#### `SystemMessage` to string

If using `SystemMessage` objects in the system prompt, extract the string content:

<CodeGroup>
```python v1 (new)
from langchain.agents import create_agent

agent = create_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[check_weather],
    system_prompt="You are a helpful assistant"  # [!code highlight]
)
```
```python v0 (old)
from langchain.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[check_weather],
    prompt=SystemMessage(content="You are a helpful assistant")  # [!code highlight]
)
```
</CodeGroup>

#### Dynamic prompts

Dynamic prompts are a core context engineering pattern— they adapt what you tell the model based on the current conversation state. To do this, use the `@dynamic_prompt` decorator:

<CodeGroup>
```python v1 (new)
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.runtime import Runtime

@dataclass
class Context:  # [!code highlight]
    user_role: str = "user"

@dynamic_prompt  # [!code highlight]
def dynamic_prompt(request: ModelRequest) -> str:  # [!code highlight]
    user_role = request.runtime.context.user_role
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        prompt = (
            f"{base_prompt} Provide detailed technical responses."
        )
    elif user_role == "beginner":
        prompt = (
            f"{base_prompt} Explain concepts simply and avoid jargon."
        )
    else:
        prompt = base_prompt

    return prompt  # [!code highlight]

agent = create_agent(
    model="openai:gpt-4o",
    tools=tools,
    middleware=[dynamic_prompt],  # [!code highlight]
    context_schema=Context
)

# Use with context
agent.invoke(
    {"messages": [{"role": "user", "content": "Explain async programming"}]},
    context=Context(user_role="expert")
)
```

```python v0 (old)
from dataclasses import dataclass

from langgraph.prebuilt import create_react_agent, AgentState
from langgraph.runtime import get_runtime

@dataclass
class Context:
    user_role: str

def dynamic_prompt(state: AgentState) -> str:
    runtime = get_runtime(Context)  # [!code highlight]
    user_role = runtime.context.user_role
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."
    return base_prompt

agent = create_react_agent(
    model="openai:gpt-4o",
    tools=tools,
    prompt=dynamic_prompt,
    context_schema=Context
)

# Use with context
agent.invoke(
    {"messages": [{"role": "user", "content": "Explain async programming"}]},
    context=Context(user_role="expert")
)
```
</CodeGroup>


### Pre-model hook

Pre-model hooks are now implemented as middleware with the `before_model` method.
This new pattern is more extensible-- you can define multiple middlewares to run before the model is called,
reusing common patterns across different agents.

Common use cases include:
* Summarizing conversation history
* Trimming messages
* Input guardrails, like PII redaction

v1 now has summarization middleware built in:

<CodeGroup>
```python v1 (new)
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=tools,
    middleware=[
        SummarizationMiddleware(  # [!code highlight]
            model="anthropic:claude-sonnet-4-5-20250929",  # [!code highlight]
            max_tokens_before_summary=1000  # [!code highlight]
        )  # [!code highlight]
    ]  # [!code highlight]
)
```
```python v0 (old)
from langgraph.prebuilt import create_react_agent, AgentState

def custom_summarization_function(state: AgentState):
    """Custom logic for message summarization."""
    ...

agent = create_react_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=tools,
    pre_model_hook=custom_summarization_function
)
```
</CodeGroup>

### Post-model hook

Post-model hooks are now implemented as middleware with the `after_model` method.
This new pattern is more extensible-- you can define multiple middlewares to run after the model is called,
reusing common patterns across different agents.

Common use cases include:
* Human in the loop
* Output guardrails

v1 has a built in middleware for human in the loop approval for tool calls:

<CodeGroup>
```python v1 (new)
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

agent = create_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[read_email, send_email],
    middleware=[HumanInTheLoopMiddleware(
        interrupt_on={
            "send_email": True,
            "description": "Please review this email before sending"
        },
    )]
)
```

```python v0 (old)
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import AgentState

def custom_human_in_the_loop_hook(state: AgentState):
    """Custom logic for human in the loop approval."""
    ...

agent = create_react_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[read_email, send_email],
    post_model_hook=custom_human_in_the_loop_hook
)
```
</CodeGroup>

### Custom state

Custom state is now defined in middleware using the `state_schema` attribute:

<CodeGroup>
```python v1 (new)
from typing import Annotated
from langchain.tools import tool
from langchain.agents import create_agent  # [!code highlight]
from langchain.agents.middleware import AgentMiddleware, AgentState  # [!code highlight]
from langgraph.prebuilt import InjectedState

# Define custom state extending AgentState
class CustomState(AgentState):
    user_name: str

# Create middleware that manages custom state
class UserStateMiddleware(AgentMiddleware[CustomState]):  # [!code highlight]
    state_schema = CustomState  # [!code highlight]

@tool  # [!code highlight]
def greet(
    state: Annotated[CustomState, InjectedState]
) -> str:
    """Use this to greet the user by name."""
    user_name = state.get("user_name", "Unknown")  # [!code highlight]
    return f"Hello {user_name}!"

agent = create_agent(  # [!code highlight]
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[greet],
    middleware=[UserStateMiddleware()]  # [!code highlight]
)
```
```python v0 (old)
from typing import Annotated
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

class CustomState(AgentState):
    user_name: str

def greet(
    state: Annotated[CustomState, InjectedState]
) -> str:
    """Use this to greet the user by name."""
    user_name = state["user_name"]
    return f"Hello {user_name}!"

agent = create_react_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[greet],
    state_schema=CustomState
)
```
</CodeGroup>

<Note>
    Custom state is defined by creating a class that extends `AgentState` and assigning it to the middleware's `state_schema` attribute.
</Note>

#### State type restrictions

`create_agent` now only supports `TypedDict` for state schemas. Pydantic models and dataclasses are no longer supported.

<CodeGroup>
```python v1 (new)
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware

# AgentState is a TypedDict
class CustomAgentState(AgentState):  # [!code highlight]
    user_id: str

class CustomAgentMiddleware(AgentMiddleware[CustomAgentState]):
    state_schema = CustomAgentState

agent = create_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=tools,
    middleware=[CustomAgentMiddleware()]
)
```

```python v0 (old)
from typing_extensions import Annotated

from pydantic import BaseModel
from langgraph.graph import StateGraph
from langgraph.graph.messages import add_messages
from langchain_core.messages import AnyMessage


class AgentState(BaseModel):  # [!code highlight]
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str

agent = create_react_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=tools,
    state_schema=AgentState
)
```
</CodeGroup>

Simply inherit from `langchain.agents.AgentState` instead of `BaseModel` or decorating with `dataclass`.
If you need to perform validation, handle it in middleware hooks instead.

### Model

Dynamic model selection allows you to choose different models based on runtime context (e.g., task complexity, cost constraints, or user preferences). `create_react_agent` released in v0.6 of [`langgraph-prebuilt`](https://pypi.org/project/langgraph-prebuilt) supported dynamic model and tool selection
via a callable passed to the `model` parameter.

This functionality has been ported to the middleware interface in v1.

#### Dynamic model selection

<CodeGroup>
```python v1 (new)
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware, ModelRequest, ModelRequestHandler
)
from langchain.messages import AIMessage
from langchain_openai import ChatOpenAI

basic_model = ChatOpenAI(model="gpt-5-nano")
advanced_model = ChatOpenAI(model="gpt-5")

class DynamicModelMiddleware(AgentMiddleware):

    def wrap_model_call(self, request: ModelRequest, handler: ModelRequestHandler) -> AIMessage:
        if len(request.state.messages) > self.messages_threshold:
            model = advanced_model
        else:
            model = basic_model

        return handler(request.replace(model=model))

    def __init__(self, messages_threshold: int) -> None:
        self.messages_threshold = messages_threshold

agent = create_agent(
    model=basic_model,
    tools=tools,
    middleware=[DynamicModelMiddleware(messages_threshold=10)]
)
```
```python v0 (old)
from langgraph.prebuilt import create_react_agent, AgentState
from langchain_openai import ChatOpenAI

basic_model = ChatOpenAI(model="gpt-5-nano")
advanced_model = ChatOpenAI(model="gpt-5")

def select_model(state: AgentState) -> BaseChatModel:
    # use a more advanced model for longer conversations
    if len(state.messages) > 10:
        return advanced_model
    return basic_model

agent = create_react_agent(
    model=select_model,
    tools=tools,
)
```
</CodeGroup>

#### Pre-bound models

To better support structured output, `create_agent` no longer accepts pre-bound models with tools or configuration:

```python
# No longer supported
model_with_tools = ChatOpenAI().bind_tools([some_tool])
agent = create_agent(model_with_tools, tools=[])

# Use instead
agent = create_agent("openai:gpt-4o-mini", tools=[some_tool])
```

<Note>
Dynamic model functions can return pre-bound models if structured output is *not* used.
</Note>

### Tools

The `tools` argument to `create_agent` accepts a list of:

* LangChain `BaseTool` instances (functions decorated with `@tool`)
* Callable objects (functions) with proper type hints and a docstring
* `dict` that represents a built-in provider tools

It no longer accepts `ToolNode` instances.

<CodeGroup>
```python v1 (new)
from langchain.agents import create_agent

agent = create_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[check_weather, search_web]
)
```
```python v0 (old)
from langgraph.prebuilt import create_react_agent, ToolNode

agent = create_react_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=ToolNode([check_weather, search_web]) # [!code highlight]
)
```
</CodeGroup>

#### Handling tool errors

You can now configure the handling of tool errors with middleware implementing the `wrap_tool_call` method.

<CodeGroup>
```python v1 (new)
# Example coming soon
```
```python v0 (old)
# Example coming soon
```
</CodeGroup>

### Structured output

#### Node changes

Structured output used to be generated in a separate node from the main agent. This is no longer the case.
We generate structured output in the main loop, reducing cost and latency.

#### Tool and provider strategies

In v1, there are two new structured output strategies:

* `ToolStrategy` uses artificial tool calling to generate structured output
* `ProviderStrategy` uses provider-native structured output generation

<CodeGroup>
```python v1 (new)
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from pydantic import BaseModel

class OutputSchema(BaseModel):
    summary: str
    sentiment: str

# Using ToolStrategy
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=tools,
    # explicitly using tool strategy
    response_format=ToolStrategy(OutputSchema)  # [!code highlight]
)
```

```python v0 (old)
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

class OutputSchema(BaseModel):
    summary: str
    sentiment: str

agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=tools,
    # using tool strategy by default with no option for provider strategy
    response_format=OutputSchema  # [!code highlight]
)

# OR

agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=tools,
    # using a custom prompt to instruct the model to generate the output schema
    response_format=("please generate ...", OutputSchema)  # [!code highlight]
)
```
</CodeGroup>

#### Prompted output removed

**Prompted output** is no longer supported via the `response_format` argument. Compared to strategies
like artificial tool calling and provider native structured output, prompted output has not proven to be particularly reliable.

### Streaming node name rename

When streaming events from agents, the node name has changed from `"agent"` to `"model"` to better reflect the node's purpose.

{/* TODO: add diagrams */}

### Runtime context

When you invoke an agent, it's often the case that you want to pass two types of data:
* Dynamic state that changes throughout the conversation (e.g., message history)
* Static context that doesn't change during the conversation (e.g., user metadata)

In v1, static context is supported by setting the `context` parameter to `invoke` and `stream`.

<CodeGroup>
```python v1 (new)
from dataclasses import dataclass

from langchain.agents import create_agent

@dataclass
class Context:
    user_id: str
    session_id: str

agent = create_agent(
    model=model,
    tools=tools,
    context_schema=ContextSchema  # [!code highlight]
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    context=Context(user_id="123", session_id="abc")  # [!code highlight]
)
```
```python v0 (old)
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, tools)

# Pass context via configurable
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    config={  # [!code highlight]
        "configurable": {  # [!code highlight]
            "user_id": "123",  # [!code highlight]
            "session_id": "abc"  # [!code highlight]
        }  # [!code highlight]
    }  # [!code highlight]
)
```
</CodeGroup>

<Note>
    The old `config["configurable"]` pattern still works for backward compatibility, but using the new `context` parameter is recommended for new applications or applications migrating to v1.
</Note>

---

## Breaking changes

### Dropped Python 3.9 support

All LangChain packages now require **Python 3.10 or higher**. Python 3.9 reaches [end of life](https://devguide.python.org/versions/) in October 2025.

### Updated return type for chat models

The return type signature for chat model invocation has been fixed from `BaseMessage` to `AIMessage`. Custom chat models implementing `bind_tools` should update their return signature:

<CodeGroup>
```python v1 (new)
Runnable[LanguageModelInput, AIMessage]
```
```python v0 (old)
Runnable[LanguageModelInput, BaseMessage]
```
</CodeGroup>

### Default message format for OpenAI Responses API

When interacting with the Responses API, `langchain-openai` now defaults to storing response items in message `content`. To restore previous behavior, set the `LC_OUTPUT_VERSION` environment variable to `v0`, or specify `output_version="v0"` when instantiating `ChatOpenAI`.

```python
# Enforce previous behavior with output_version flag
model = ChatOpenAI(model="gpt-4o-mini", output_version="v0")
```

### Default `max_tokens` in `langchain-anthropic`

The `max_tokens` parameter now defaults to higher values based on the model chosen, rather than the previous default of `1024`. If you relied on the old default, explicitly set `max_tokens=1024`.

### Legacy code moved to `langchain-classic`

Existing functionality outside the focus of standard interfaces and agents has been moved to the [`langchain-classic`](https://pypi.org/project/langchain-classic) package. See the [Simplified namespace](#simplified-namespace) section for details on what's available in the core `langchain` package and what moved to `langchain-classic`.

### Removal of deprecated APIs

Methods, functions, and other objects that were already deprecated and slated for removal in 1.0 have been deleted. Check the [deprecation notices](https://python.langchain.com/docs/versions/migrating_chains) from previous versions for replacement APIs.

### `.text()` is now a property

Use of the `.text()` method on message objects should drop the parentheses:

<CodeGroup>
```python v1 (new)
# Property access
text = response.text

# deprecated method call
text = response.text()
```
```python v0 (old)
text = response.text()
```
</CodeGroup>

Existing usage patterns (i.e., `.text()`) will continue to function but now emit a warning.
</file>

<file path="What's new in v1.mdx">
---
title: What's new in v1
sidebarTitle: What's new
---

import AlphaCallout from '/snippets/alpha-lc-callout.mdx';

<AlphaCallout />

**LangChain v1 is a focused, production-ready foundation for building agentic applications.** We've streamlined the framework around three core improvements:

<CardGroup cols={1}>
    <Card title="create_agent" icon="robot" href="#create-agent" arrow>
        A new standard way to build agents in LangChain, replacing `langgraph.prebuilt.create_react_agent` with a cleaner, more powerful API.
    </Card>
    <Card title="Standard content blocks" icon="cube" href="#standard-content-blocks" arrow>
        A new `content_blocks` property that provides unified access to modern LLM features across all providers.
    </Card>
    <Card title="Simplified namespace" icon="sitemap" href="#simplified-namespace" arrow>
        The `langchain` namespace has been streamlined to focus on essential building blocks for agents, with legacy functionality moved to `langchain-classic`.
    </Card>
</CardGroup>


## `create_agent`

`create_agent` is the standard way to build agents in LangChain 1.0. It provides a simpler interface than `langgraph.prebuilt.create_react_agent` while offering greater customization potential by using middleware.

```python
from langchain.agents import create_agent

agent = create_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[search_web, analyze_data, send_email],
    system_prompt="You are a helpful research assistant."
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Research AI safety trends"}
    ]
})
```

Under the hood, `create_agent` is built on the basic agent loop -- calling a model, letting it choose tools to execute, and then finishing when it calls no more tools:

<div style={{ display: "flex", justifyContent: "center" }}>
  <img
    src="/oss/images/core_agent_loop.png"
    alt="Core agent loop diagram"
    height="300"
    className="rounded-lg"
  />
</div>

For more information, see [Agents](/oss/langchain/agents).

### Middleware

Middleware is the defining feature of `create_agent`. It makes `create_agent` highly customizable, raising the ceiling for what you can build.

Great agents require [context engineering](/oss/langchain/context-engineering): getting the right information to the model at the right time. Middleware helps you control dynamic prompts, conversation summarization, selective tool access, state management, and guardrails through a composable abstraction.

#### Prebuilt middleware

LangChain provides a few prebuilt middlewares for common patterns, including:

- `PIIRedactionMiddleware`: Redact sensitive information before sending to the model
- `SummarizationMiddleware`: Condense conversation history when it gets too long
- `HumanInTheLoopMiddleware`: Require approval for sensitive tool calls

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    PIIRedactionMiddleware,
    SummarizationMiddleware,
    HumanInTheLoopMiddleware
)

agent = create_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[read_email, send_email],
    middleware=[
        PIIRedactionMiddleware(patterns=["email", "phone", "ssn"]),
        SummarizationMiddleware(
            model="anthropic:claude-sonnet-4-5-20250929",
            max_tokens_before_summary=500
        ),
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                }
            }
        ),
    ]
)
```

#### Custom middleware

You can also build custom middleware to fit your specific needs. Middleware exposes hooks at each step in an agent's execution:

<div style={{ display: "flex", justifyContent: "center" }}>
  <img
    src="/oss/images/middleware_final.png"
    alt="Middleware flow diagram"
    height="300"
    className="rounded-lg"
  />
</div>

Build custom middleware by implementing any of these hooks on a subclass of the `AgentMiddleware` class:

| Hook              | When it runs             | Use cases                               |
|-------------------|--------------------------|-----------------------------------------|
| `before_agent`    | Before calling the agent | Load memory, validate input             |
| `before_model`    | Before each LLM call     | Update prompts, trim messages           |
| `wrap_model_call` | Around each LLM call     | Intercept and modify requests/responses |
| `wrap_tool_call`  | Around each tool call    | Intercept and modify tool execution     |
| `after_model`     | After each LLM response  | Validate output, apply guardrails       |
| `after_agent`     | After agent completes    | Save results, cleanup                   |


Example custom middleware:

```python expandable
from dataclasses import dataclass

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelRequestHandler
)
from langchain.messages import AIMessage

@dataclass
class Context:
    user_expertise: str = "beginner"

class ExpertiseBasedToolMiddleware(Middleware):
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: ModelRequestHandler
    ) -> AIMessage:
        user_level = request.runtime.context.user_expertise

        if user_level == "expert":
            # More powerful model
            model = "openai:gpt-5"
            tools = [advanced_search, data_analysis]
        else:
            # Less powerful model
            model = "openai:gpt-5-nano"
            tools = [simple_search, basic_calculator]

        return handler(
            request.replace(model=model, tools=tools)
        )

agent = create_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[
        simple_search,
        advanced_search,
        basic_calculator,
        data_analysis
    ],
    middleware=[ExpertiseBasedToolMiddleware()],
    context_schema=Context
)
```

For more information, see [the complete middleware guide](/oss/langchain/middleware).

### Built on LangGraph

Because `create_agent` is built on LangGraph, you automatically get built in support
for long running, reliable agents via:

<CardGroup cols={2}>
    <Card title="Persistence" icon="database">
        Conversations automatically persist across sessions with built-in checkpointing
    </Card>
    <Card title="Streaming" icon="water">
        Stream tokens, tool calls, and reasoning traces in real-time
    </Card>
    <Card title="Human-in-the-loop" icon="hand">
        Pause agent execution for human approval before sensitive actions
    </Card>
    <Card title="Time travel" icon="clock-rotate-left">
        Rewind conversations to any point and explore alternate paths and prompts
    </Card>
</CardGroup>

You don't need to learn LangGraph to use these features—they work out of the box.

### Structured output

`create_agent` has improved structured output generation:

- **Main loop integration**: Structured output is now generated in the main loop instead of requiring an additional LLM call
- **Structured output strategy**: Models can choose between calling tools or using provider-side structured output generation
- **Cost reduction**: Eliminates extra expense from additional LLM calls

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel

class Weather(BaseModel):
    temperature: float
    condition: str

def weather_tool(city: str) -> str:
    """Get the weather for a city."""
    return f"it's sunny and 70 degrees in {city}"

agent = create_agent(
    "openai:gpt-4o-mini",
    tools=[weather_tool],
    response_format=ToolStrategy(Weather)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in SF?"}]
})

print(repr(result["structured_response"]))
# results in `Weather(temperature=70.0, condition='sunny')`
```

**Error handling**: Control error handling via the `handle_errors` parameter to `ToolStrategy`:
- **Parsing errors**: Model generates data that doesn't match desired structure
- **Multiple tool calls**: Model generates 2+ tool calls for structured output schemas

---

## Standard content blocks

<Note>
    1.0 Alpha releases are available for most packages. Only the following currently support new content blocks:

    - [`langchain`](https://pypi.org/project/langchain/)
    - [`langchain-core`](https://pypi.org/project/langchain-core/)
    - [`langchain-anthropic`](https://pypi.org/project/langchain-anthropic/)
    - [`langchain-aws`](https://pypi.org/project/langchain-aws/)
    - [`langchain-openai`](https://pypi.org/project/langchain-openai/)
    - [`langchain-google-genai`](https://pypi.org/project/langchain-google-genai/)
    - [`langchain-ollama`](https://pypi.org/project/langchain-ollama/)

    Broader support for content blocks will be rolled out during the alpha period and following stable release.
</Note>

The new `content_blocks` property provides unified access to modern LLM features across all providers:

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
response = model.invoke("What's the capital of France?")

# Unified access to content blocks
for block in response.content_blocks:
    if block["type"] == "reasoning":
        print(f"Model reasoning: {block['reasoning']}")
    elif block["type"] == "text":
        print(f"Response: {block['text']}")
    elif block["type"] == "tool_call":
        print(f"Tool call: {block['name']}({block['args']})")
```

### Benefits

- **Provider agnostic**: Access reasoning traces, citations, built-in tools (web search, code interpreters, etc.), and other features using the same API regardless of provider
- **Type safe**: Full type hints for all content block types
- **Backward compatible**: Standard content can be [loaded lazily](/oss/langchain/messages#standard-content-blocks), so there are no associated breaking changes

For more information, see our guide on [content blocks](/oss/langchain/messages#content)

---

## Simplified package

LangChain v1 streamlines the `langchain` package namespace to focus on essential building blocks for agents. The namespace exposes only the most useful and relevant functionality:

### Namespace

The v1 namespace includes:

| Module | What's available | Notes |
|--------|------------------|-------|
| `langchain.agents` | `create_agent`, `AgentState` | Core agent creation functionality |
| `langchain.messages` | Message types, content blocks, `trim_messages` | Re-exported from `langchain-core` |
| `langchain.tools` | `tool`, `BaseTool`, injection helpers | Re-exported from `langchain-core` |
| `langchain.chat_models` | `init_chat_model`, `BaseChatModel` | Unified model initialization |
| `langchain.embeddings` | `Embeddings`, `init_embeddings` | Embedding models |

Most of these exports are re-exported from `langchain-core` for convenience, giving you a focused API surface for building agents.

```python
# Agent building
from langchain.agents import create_agent

# Messages and content
from langchain.messages import AIMessage, HumanMessage

# Tools
from langchain.tools import tool

# Model initialization
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
```

### `langchain-classic`

Legacy functionality has moved to [`langchain-classic`](https://pypi.org/project/langchain-classic) to keep the core package lean and focused.

#### What's in `langchain-classic`

- Legacy chains and chain implementations
- The indexing API
- [`langchain-community`](https://pypi.org/project/langchain-community) exports
- Other deprecated functionality

If you use any of this functionality, install [`langchain-classic`](https://pypi.org/project/langchain-classic):

<CodeGroup>
```bash pip
pip install langchain-classic
```

```bash uv
uv add langchain-classic
```
</CodeGroup>

Then update your imports:

```python
from langchain import ...  # [!code --]
from langchain_classic import ...  # [!code ++]

from langchain.chains import ...  # [!code --]
from langchain_classic.chains import ...  # [!code ++]
```

## Reporting issues

Please report any issues discovered with 1.0 on [GitHub](https://github.com/langchain-ai/langchain/issues) using the `'v1'` [label](https://github.com/langchain-ai/langchain/issues?q=state%3Aopen%20label%3Av1).

## Additional resources

<CardGroup cols={3}>
    <Card title="LangChain 1.0" icon="rocket" href="https://blog.langchain.com/langchain-langchain-1-0-alpha-releases/">
        Read the announcement
    </Card>
    <Card title="Middleware Guide" icon="puzzle-piece" href="https://blog.langchain.com/agent-middleware/">
        Deep dive into middleware
    </Card>
    <Card title="Agents Documentation" icon="book" href="/oss/langchain/agents" arrow>
        Full agent documentation
    </Card>
    <Card title="Message Content" icon="message" href="/oss/langchain/messages#content" arrow>
        New content blocks API
    </Card>
    <Card title="Migration guide" icon="arrow-right-arrow-left" href="/oss/migrate/langchain-v1" arrow>
        How to migrate to LangChain v1
    </Card>
    <Card title="GitHub" icon="github" href="https://github.com/langchain-ai/langchain">
        Report issues or contribute
    </Card>
</CardGroup>

## See also

- [Versioning](/oss/versioning) - Understanding version numbers
- [Release policy](/oss/release-policy) - Detailed release policies
</file>

</files>
