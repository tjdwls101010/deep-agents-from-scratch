WRITE_TODOS_DESCRIPTION = """Use this tool to create and manage a structured task list for your current work session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.

## When to Use This Tool
Use this tool proactively in these scenarios:
1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done

## Task States
- pending: Task not yet started
- in_progress: Currently working on (limit to ONE task at a time)
- completed: Task finished successfully"""

FILE_TOOLS_DESCRIPTION = """Use these tools to manage a virtual file system stored in your agent state. This enables context offloading, code organization, and persistent memory across conversations.

## Core File Operations

**ls()**: List all files in the virtual filesystem
- Shows what files currently exist in agent memory
- Use to orient yourself before other file operations

**read_file(file_path, offset=0, limit=2000)**: Read file content with pagination
- Returns content with line numbers (like `cat -n`)  
- Use offset/limit for large files to avoid context overflow
- Essential before making any edits to understand existing content

**write_file(file_path, content)**: Create or completely overwrite a file
- Creates new files or replaces entire file contents
- Use for initial file creation or complete rewrites
- Stores file persistently in agent state

**edit_file(file_path, old_string, new_string, replace_all=False)**: Precise string replacement
- Requires exact string matching including whitespace and indentation
- Fails if old_string appears multiple times (unless replace_all=True)
- Use replace_all for systematic renaming across a file
- Always read the file first to ensure exact string matching

## Usage Patterns

**Context Management**: Use files to offload information from conversation context, enabling longer-running tasks without hitting context limits.

**Code Organization**: Store related code snippets, configurations, and documentation in organized file structures.

**Memory Persistence**: Files survive across tool calls within the same conversation, providing persistent memory.

**Exact String Matching**: When editing, copy the exact text including spaces, tabs, and line breaks from read_file output to ensure successful edits.

## Best Practices

1. Always read files before editing to understand current content
2. Use descriptive file paths that reflect content purpose  
3. Break large files into smaller, focused files when possible
4. Leverage the virtual nature - experiment freely without affecting real filesystem
5. Use ls() frequently to maintain awareness of your file organization""" 

TASK_DESCRIPTION_PREFIX = """Launch a new agent to handle complex, multi-step tasks autonomously. 

Available agent types and the tools they have access to:
- general-purpose: General-purpose agent for researching complex questions, searching for files and content, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you. (Tools: *)
{other_agents}
"""

TASK_DESCRIPTION_SUFFIX = """When using the Task tool, you must specify a subagent_type parameter to select which agent type to use.

When to use the Agent tool:
- When you are instructed to execute custom slash commands. Use the Agent tool with the slash command invocation as the entire prompt. The slash command can take arguments. For example: Task(description="Check the file", prompt="/check-file path/to/file.py")

When NOT to use the Agent tool:
- If you want to read a specific file path, use the Read or Glob tool instead of the Agent tool, to find the match more quickly
- If you are searching for a specific term or definition within a known location, use the Glob tool instead, to find the match more quickly
- If you are searching for content within a specific file or set of 2-3 files, use the Read tool instead of the Agent tool, to find the match more quickly
- Other tasks that are not related to the agent descriptions above


Usage notes:
1. Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses
2. When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
3. Each agent invocation is stateless. You will not be able to send additional messages to the agent, nor will the agent be able to communicate with you outside of its final report. Therefore, your prompt should contain a highly detailed task description for the agent to perform autonomously and you should specify exactly what information the agent should return back to you in its final and only message to you.
4. The agent's outputs should generally be trusted
5. Clearly tell the agent whether you expect it to create content, perform analysis, or just do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent
6. If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.

Example usage:

<example_agent_descriptions>
"content-reviewer": use this agent after you are done creating significant content or documents
"greeting-responder": use this agent when to respond to user greetings with a friendly joke
"research-analyst": use this agent to conduct thorough research on complex topics
</example_agent_description>

<example>
user: "Please write a function that checks if a number is prime"
assistant: Sure let me write a function that checks if a number is prime
assistant: First let me use the Write tool to write a function that checks if a number is prime
assistant: I'm going to use the Write tool to write the following code:
<code>
function isPrime(n) {
  if (n <= 1) return false
  for (let i = 2; i * i <= n; i++) {
    if (n % i === 0) return false
  }
  return true
}
</code>
<commentary>
Since significant content was created and the task was completed, now use the content-reviewer agent to review the work
</commentary>
assistant: Now let me use the content-reviewer agent to review the code
assistant: Uses the Task tool to launch with the content-reviewer agent 
</example>

<example>
user: "Can you help me research the environmental impact of different renewable energy sources and create a comprehensive report?"
<commentary>
This is a complex research task that would benefit from using the research-analyst agent to conduct thorough analysis
</commentary>
assistant: I'll help you research the environmental impact of renewable energy sources. Let me use the research-analyst agent to conduct comprehensive research on this topic.
assistant: Uses the Task tool to launch with the research-analyst agent, providing detailed instructions about what research to conduct and what format the report should take
</example>

<example>
user: "Hello"
<commentary>
Since the user is greeting, use the greeting-responder agent to respond with a friendly joke
</commentary>
assistant: "I'm going to use the Task tool to launch with the greeting-responder agent"
</example>"""