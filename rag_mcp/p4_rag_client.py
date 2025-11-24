import asyncio
import uuid
from typing import AsyncGenerator
import ollama
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import json

# =========================
# Ollama + MCP Client
# =========================
class OllamaMCPClient:
    def __init__(self, model: str, server_url: str):
        self.model = model
        self.server_url = server_url
        self.messages: list[dict] = []
        self.available_tools: list[dict] = []

        self.system_prompt = (
            "You are a UI automation assistant using MCP Agent with dynamic workflows.\n\n"

            "## Available Tools:\n\n"

            "1. **click_number(number)** - Simple workflow when you know the exact number\n"
            "   Use when: User says 'click 5' or provides specific number\n"
            "   Example: click_number('5')\n\n"

            "2. **navigate_to(target)** - Standard workflow to find and click an element\n"
            "   Use when: User wants to click a specific element by name\n"
            "   Example: navigate_to('home') for 'go to home'\n"
            "   Example: navigate_to('settings') for 'open settings'\n\n"

            "3. **translate_vc(command)** - Single voice command translation\n"
            "   Use with /VC: prefix\n\n"

            "4. **translate_cc(command)** - Direct voice command\n"
            "   Use with /CC: prefix\n\n"

            "## Command Patterns:\n\n"

            "**/VC: <command>** - Translate natural language\n"
            "Example: /VC: click home → translate_vc('click home')\n\n"

            "**/CC: <command>** - Direct command\n"
            "Example: /CC: CLICK 5 → translate_cc('CLICK 5')\n\n"

            "**/exe: <task>** - Multi-step automation (choose appropriate workflow)\n"
            "Examples:\n"
            "- /exe: go to home → navigate_to('home')\n"
            "- /exe: click 5 → click_number('5')\n"
            "- /exe: go to settings → navigate_to('settings')\n\n"

            "**No prefix** - Normal conversation\n\n"

            "## Decision Logic:\n\n"

            "For /exe: commands, analyze the task:\n"
            "- If user mentions specific number → use click_number()\n"
            "- If user wants single element → use navigate_to()\n"
            "- If user wants multiple steps → use navigate_multi_step()\n\n"

            "## Important:\n"
            "- Each tool handles the ENTIRE workflow automatically\n"
            "- You only need to call ONE tool per /exe: command\n"
            "- The server guarantees sequential execution via mcp-agent workflows\n"
            "- Do NOT try to break down into manual steps\n"
            "- Let the workflow tools handle all the complexity\n\n"

            "## Examples:\n\n"

            "User: /exe: go to home\n"
            "You: Call navigate_to('home') → Done\n\n"

            "User: /exe: click number 5\n"
            "You: Call click_number('5') → Done\n\n"

            "User: /exe: go to settings then click profile\n"
            "You: Call navigate_multi_step(['settings', 'profile']) → Done\n\n"

            "User: /VC: click home\n"
            "You: Call translate_vc('click home') → Return command\n\n"

            "User: Hello\n"
            "You: Hi! I can help with UI automation. Use /exe:, /VC:, or /CC: commands.\n"
        )

    async def initialize_tools(self):
        """Initialize connection and fetch tools (one-time operation)."""
        try:
            async with streamablehttp_client(url=self.server_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    response = await session.list_tools()
                    print(response)
                    self.available_tools = []
                    for tool in response.tools:
                        self.available_tools.append({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema,
                            },
                        })

                    print(f"Connected! Available tools: {[t['function']['name'] for t in self.available_tools]}")

            self.messages = [{"role": "system", "content": self.system_prompt}]
            return True, f"Connected! Tools available: {len(self.available_tools)}"
        except Exception as e:
            return False, f"Connection failed: {e}"

    def _execute_tool_sync(self, tool_call) -> str:
        """Execute tool synchronously by creating a new event loop."""

        async def _do_execute():
            async with streamablehttp_client(url=self.server_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments or {}
                    result = await session.call_tool(tool_name, tool_args)
                    if result.content and len(result.content) > 0 and getattr(result.content[0], "text", None):
                        return result.content[0].text
                    return "Tool executed but returned no content."

        # Create new event loop for each tool execution
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_do_execute())
        finally:
            loop.close()

    def chat_stream(self, user_text: str, max_tool_turns: int = 3):
        """
        Generator that streams responses.
        Reduced max_tool_turns to 3 since each workflow tool is complete.
        """
        self.messages.append({"role": "user", "content": user_text})

        for turn in range(max_tool_turns):
            stream = ollama.chat(
                model=self.model,
                messages=self.messages,
                tools=self.available_tools,
                stream=True,
            )

            assistant_text = ""
            tool_calls = []

            # Stream tokens as they arrive
            for chunk in stream:
                if chunk.message.content:
                    assistant_text += chunk.message.content
                    yield chunk.message.content
                if chunk.message.tool_calls:
                    tool_calls.extend(chunk.message.tool_calls)

            self.messages.append(
                {"role": "assistant", "content": assistant_text, "tool_calls": tool_calls or []}
            )

            if not tool_calls:
                return

            # Execute tools synchronously
            yield f"\n\n🔧 Executing workflow tool...\n\n"

            for tc in tool_calls:
                tool_name = tc.function.name
                tool_args = tc.function.arguments or {}

                yield f"🔹 **Workflow:** {tool_name}\n"
                yield f"   **Args:** {json.dumps(tool_args, indent=2)}\n\n"

                try:
                    raw = self._execute_tool_sync(tc)
                    tool_call_id = str(uuid.uuid4())

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": raw
                    })

                    # Parse result to show summary
                    try:
                        result_json = json.loads(raw)
                        if 'status' in result_json:
                            status = result_json.get('status', 'unknown')
                            if status == 'completed' or status == 'success':
                                yield f"   ✅ **Status:** Workflow completed successfully\n"
                                if 'message' in result_json:
                                    yield f"   📝 **Result:** {result_json['message']}\n"
                            else:
                                yield f"   ⚠️ **Status:** {status}\n"

                        # Show steps if available
                        if 'steps' in result_json:
                            steps = result_json['steps']
                            if isinstance(steps, list):
                                yield f"   📊 **Steps executed:** {len(steps)}\n"

                    except json.JSONDecodeError:
                        # If not JSON, show preview
                        preview = raw[:300] + "..." if len(raw) > 300 else raw
                        yield f"   📄 **Result:** {preview}\n"

                except Exception as e:
                    yield f"   ❌ **Error:** {e}\n"

            yield "\n---\n\n"

            # Stream follow-up response
            stream2 = ollama.chat(
                model=self.model,
                messages=self.messages,
                tools=self.available_tools,
                stream=True,
            )

            followup_text = ""
            followup_tool_calls = []

            for chunk in stream2:
                if chunk.message.content:
                    followup_text += chunk.message.content
                    yield chunk.message.content
                if chunk.message.tool_calls:
                    followup_tool_calls.extend(chunk.message.tool_calls)

            self.messages.append({
                "role": "assistant",
                "content": followup_text,
                "tool_calls": followup_tool_calls or []
            })

            # If no more tool calls, we're done
            if not followup_tool_calls:
                return

        yield "\n\n⚠️ Reached maximum turns. Task may be incomplete."