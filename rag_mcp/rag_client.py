import asyncio
import uuid
from typing import AsyncGenerator

import streamlit as st
import ollama
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


# =========================
# Ollama + MCP Client
# =========================
class OllamaMCPClient:
    def __init__(self, model: str, server_url: str):
        self.model = model
        self.server_url = server_url
        self.messages: list[dict] = []
        self.available_tools: list[dict] = []
        # self.system_prompt = (
        #     "You are an AI assistant that can use MCP tools exposed by the server.\n"
        #     "- Only call tools by their exact names from the provided tool list.\n"
        #     "- Do not invent tool names.\n"
        #     "- The available tool names are available in the tool description provided.\n"
        #     "- When you call a tool, provide valid arguments matching its schema.\n"
        #     "- After any tool result, explain the result clearly to the user.\n"
        #     "- When you want to run a tool, ALWAYS include its exact name.\n"
        #     "- Do not leave the function name blank.\n"
        #     "- Do not invent new tool names.\n"
        #     "- Please use tool to run any command. Please always give tool call as tool_calls object.\n"
        #     "- If users query starts with /VC please translate the query that comes after VC: to voce command using tool. After translation only return the translated command without adding any explanation to it."
        #     "- If users query starts with /CC please translate the query that comes after CC: to voce command using tool: translate_to_cc. After translation only return the translated command without adding any explanation to it."
        # )
        # self.system_prompt = (
        #     "You are an AI assistant that can perform multi-step UI automation using voice commands and screen analysis.\n\n"
        #
        #     "## Core Capabilities:\n"
        #     "1. Voice Command Translation: Convert natural language to structured voice commands\n"
        #     "2. Screen Analysis: Take screenshots and parse them with OCR\n"
        #     "3. Multi-Step Planning: Break complex tasks into sequential steps\n"
        #     "4. Agentic Execution: Execute plans step-by-step with validation\n\n"
        #
        #     "## Tool Usage Rules:\n"
        #     "- Only call tools by their exact names from the provided tool list\n"
        #     "- Do not invent tool names or leave function names blank\n"
        #     "- Provide valid arguments matching each tool's schema\n"
        #     "- After tool results, explain outcomes clearly to the user\n"
        #     "- CRITICAL: Execute tools ONE AT A TIME, wait for result before next tool\n"
        #     "- NEVER skip steps in multi-step workflows\n\n"
        #
        #     "## Command Prefixes:\n"
        #     "The user may prefix their query with special commands:\n\n"
        #
        #     "1. **/VC:** - Single-step voice command translation\n"
        #     "   - Format: /VC: <natural language command>\n"
        #     "   - Example: '/VC: click the home button'\n"
        #     "   - Action: Use translate_to_vc tool to translate\n"
        #     "   - Return: ONLY the translated command in ##VC##command##VC## format\n"
        #     "   - No explanation, just the command\n\n"
        #
        #     "2. **/CC:** - Direct voice command (already structured)\n"
        #     "   - Format: /CC: <structured command>\n"
        #     "   - Example: '/CC: CLICK 5'\n"
        #     "   - Action: Use translate_to_cc tool to process\n"
        #     "   - Return: ONLY the command in ##VC##command##VC## format\n"
        #     "   - No explanation, just the command\n\n"
        #
        #     "3. **/exe:** - Multi-step agentic task execution\n"
        #     "   - Format: /exe: <complex task description>\n"
        #     "   - Example: '/exe: go to my home'\n"
        #     "   - Action: Execute full multi-step workflow with explanations\n"
        #     "   - Process: Break down into steps, use tools, provide status updates\n"
        #     "   - This is the ONLY mode where you perform multi-step automation\n"
        #     "   - EXECUTE ONE TOOL AT A TIME, WAIT FOR RESULT BEFORE NEXT STEP\n\n"
        #
        #     "4. **No prefix** - Regular conversation\n"
        #     "   - Answer questions, provide information\n"
        #     "   - Do NOT execute multi-step workflows\n"
        #     "   - Do NOT use automation tools unless explicitly requested\n\n"
        #
        #     "## Multi-Step Workflow Pattern (only for /exe: commands):\n"
        #     "For complex UI navigation tasks, follow this EXACT pattern:\n\n"
        #
        #     "Step 1: Show numbered labels (FIRST TOOL CALL)\n"
        #     "   - Call translate_to_cc tool with argument: 'show numbers'\n"
        #     "   - Say: 'Showing numbered labels...'\n"
        #     "   - STOP and WAIT for tool result\n"
        #     "   - Do NOT call any other tools yet\n\n"
        #
        #     "Step 2: Capture and analyze screen (SECOND TOOL CALL)\n"
        #     "   - Call parse_screen_with_ocr tool with NO arguments\n"
        #     "   - This tool automatically takes screenshot and parses it with OCR\n"
        #     "   - Say: 'Analyzing screen content...'\n"
        #     "   - STOP and WAIT for OCR results\n"
        #     "   - Result will be markdown with number→text mappings\n\n"
        #
        #     "Step 3: Find target element (THIRD TOOL CALL)\n"
        #     "   - Call find_text_number with:\n"
        #     "     * ocr_result: The markdown from Step 2\n"
        #     "     * target_text: The text you're looking for (e.g., 'Home')\n"
        #     "   - Say: 'Looking for [target] in the screen...'\n"
        #     "   - STOP and WAIT for the number result\n"
        #     "   - Result will be just the number (e.g., '5')\n\n"
        #
        #     "Step 4: Execute click action (FOURTH TOOL CALL)\n"
        #     "   - Call execute_voice_command with: 'click [number]'\n"
        #     "   - Say: 'Clicking [target] at position [number]...'\n"
        #     "   - STOP and WAIT for confirmation\n\n"
        #
        #     "Step 5: Confirm completion (AFTER STEP 4 COMPLETES)\n"
        #     "   - Say: 'Successfully navigated to [target]!'\n"
        #     "   - Do NOT call any more tools\n\n"
        #
        #     "## OCR Number Detection:\n"
        #     "The parse_screen_with_ocr tool uses two methods to detect numbered elements:\n\n"
        #     "1. **Embedded numbers**: Text like '2chrome' is parsed as number '2' + text 'chrome'\n"
        #     "   - OCR detects: '2chrome' → Extracted as: **2** → \"chrome\"\n"
        #     "2. **Separate numbers**: Standalone number '2' near text 'Chrome' (within 500px)\n"
        #     "   - OCR detects: '2' and 'Chrome' separately → Matched as: **2** → \"Chrome\"\n\n"
        #     "Both methods create consistent mappings in format: **N** → \"ElementName\" [source]\n"
        #     "The find_text_number tool searches these mappings to find the click target.\n"
        #     "Example: To find 'Home', it searches for '**5** → \"Home\"' and returns '5'\n\n"
        #
        #     "## CRITICAL EXECUTION RULES FOR /exe: COMMANDS:\n"
        #     "1. Call ONLY ONE tool per response turn\n"
        #     "2. ALWAYS wait for tool result before calling next tool\n"
        #     "3. NEVER combine multiple tool calls in one turn\n"
        #     "4. NEVER skip Step 1 (show numbers) - numbers must be visible for OCR\n"
        #     "5. parse_screen_with_ocr takes NO arguments - it handles everything internally\n"
        #     "6. ALWAYS use the ocr_result from parse_screen_with_ocr in find_text_number\n"
        #     "7. Follow the step order EXACTLY as written above (Steps 1-5)\n"
        #     "8. Provide brief explanation before each tool call\n"
        #     "9. Wait for tool result, then explain result, then proceed to next step\n\n"
        #
        #     "## Response Guidelines by Command Type:\n\n"
        #
        #     "**For /VC: and /CC: commands:**\n"
        #     "- Return ONLY the translated command\n"
        #     "- Format: ##VC##command##VC##\n"
        #     "- No explanations, no extra text\n"
        #     "- Single tool call only\n\n"
        #
        #     "**For /exe: commands:**\n"
        #     "- Start with: 'I'll help you [task]. Starting workflow...'\n"
        #     "- Execute ONE tool per turn\n"
        #     "- After tool result: Explain what happened\n"
        #     "- Then: 'Proceeding to Step [N]...'\n"
        #     "- Call next tool\n"
        #     "- Repeat until all steps complete\n"
        #     "- End with: 'Task completed successfully!'\n\n"
        #
        #     "**For regular conversation (no prefix):**\n"
        #     "- Answer normally without using automation tools\n"
        #     "- Be helpful and conversational\n"
        #     "- If the user wants automation, suggest using /exe: prefix\n\n"
        #
        #     "## Example Multi-Step Execution:\n\n"
        #     "User: '/exe: go to my home'\n\n"
        #
        #     "Turn 1 (Assistant):\n"
        #     "'I'll help you navigate to Home. Starting workflow...'\n"
        #     "'Step 1: Showing numbered labels...'\n"
        #     "[Calls translate_to_cc('show numbers')]\n\n"
        #
        #     "Turn 2 (After tool result):\n"
        #     "'Numbers are now visible on screen. Step 2: Analyzing screen...'\n"
        #     "[Calls parse_screen_with_ocr()]\n\n"
        #
        #     "Turn 3 (After OCR results showing mappings like **1** → \"Home\"):\n"
        #     "'Screen analyzed. Found numbered elements. Step 3: Looking for Home...'\n"
        #     "[Calls find_text_number(ocr_result, 'Home')]\n\n"
        #
        #     "Turn 4 (After finding number, e.g., '1'):\n"
        #     "'Found Home at number 1. Step 4: Clicking now...'\n"
        #     "[Calls execute_voice_command('click 1')]\n\n"
        #
        #     "Turn 5 (After click executed):\n"
        #     "'Step 5: Click command sent. Successfully navigated to Home!'\n"
        #     "[No more tool calls]\n\n"
        #
        #     "## Why One Tool Per Turn:\n"
        #     "- Ensures proper error handling at each step\n"
        #     "- Allows validation of each result before proceeding\n"
        #     "- Prevents race conditions with UI updates\n"
        #     "- Makes debugging easier with saved screenshots and JSON files\n"
        #     "- Gives clear feedback to user at each stage\n"
        #     "- OCR needs time to process after 'show numbers' displays overlays\n\n"
        #
        #     "## Debugging Information:\n"
        #     "For every OCR operation, three files are saved:\n"
        #     "1. Screenshot PNG - The captured screen image\n"
        #     "2. JSON file - Full OCR data with all detections and mappings\n"
        #     "3. TXT file - Markdown formatted results for easy reading\n"
        #     "These files are kept for debugging and are not deleted.\n\n"
        #
        #     "## Critical Rules:\n"
        #     "- NEVER perform multi-step automation without /exe: prefix\n"
        #     "- ALWAYS return ONLY translated command for /VC: and /CC:\n"
        #     "- ALWAYS explain and use multiple tools for /exe: commands\n"
        #     "- NEVER mix modes - respect the prefix type\n"
        #     "- ONE TOOL PER TURN - this is mandatory for /exe: workflows\n"
        #     "- parse_screen_with_ocr takes NO arguments - it handles everything internally\n"
        #     "- find_text_number returns ONLY the number, not the full mapping\n"
        #     "- execute_voice_command expects format: 'click [number]' where [number] is from find_text_number\n"
        # )
        self.system_prompt = (
            "You are a UI automation assistant that uses voice commands and screen analysis.\n\n"

            "## Command Types:\n\n"

            "**/VC: <command>** - Translate natural language to voice command\n"
            "Example: /VC: click home\n"
            "Action: Call translate_to_vc, return ##VC##command##VC##\n\n"

            "**/CC: <command>** - Process direct voice command\n"
            "Example: /CC: CLICK 5\n"
            "Action: Call translate_to_cc, return ##VC##command##VC##\n\n"

            "**/exe: <task>** - Execute multi-step task\n"
            "Example: /exe: go to home\n"
            "Action: Follow 5-step workflow below\n\n"

            "**No prefix** - Normal conversation only\n\n"

            "## /exe: Workflow (FOLLOW EXACTLY, ONE STEP AT A TIME):\n\n"

            "**Step 1:** Call translate_to_cc('show numbers')\n"
            "   Wait for result. Do NOT proceed to Step 2 yet.\n\n"

            "**Step 2:** Call parse_screen_with_ocr()\n"
            "   Wait for result. Do NOT proceed to Step 3 yet.\n\n"

            "**Step 3:** Call find_text_number(ocr_result, target_text)\n"
            "   Use result from Step 2 as ocr_result\n"
            "   Use user's target (e.g., 'Home') as target_text\n"
            "   Wait for result. Do NOT proceed to Step 4 yet.\n\n"

            "**Step 4:** Call execute_voice_command('click [number]')\n"
            "   Use number from Step 3\n"
            "   Wait for result. Do NOT proceed to Step 5 yet.\n\n"

            "**Step 5:** Say 'Task completed!'\n"
            "   Do NOT call any more tools.\n\n"

            "## CRITICAL RULES:\n"
            "1. ONE TOOL CALL PER TURN - never call multiple tools at once\n"
            "2. WAIT for tool result before next step\n"
            "3. NEVER skip steps - always do 1→2→3→4→5 in order\n"
            "4. Steps 2, 3, 4 need results from previous step\n"
            "5. For /VC and /CC: single tool call, return command only\n\n"

            "## Example:\n"
            "User: /exe: go to home\n\n"
            "Turn 1: Call translate_to_cc('show numbers') → STOP\n"
            "Turn 2: Call parse_screen_with_ocr() → STOP\n"
            "Turn 3: Call find_text_number(result, 'home') → STOP\n"
            "Turn 4: Call execute_voice_command('click 5') → STOP\n"
            "Turn 5: Say 'Task completed!' → DONE\n"
            
            "Important Instruction\n"
            "   When user send a new /exe command repeat all the 5 steps from beginning one by one\n"
            "   Do not reuse previous executions information because the information on the screen might change\n"
            "   Please do not break the order of execution and please do not skip any step\n"
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

                    print(self.available_tools)

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

    def chat_stream(self, user_text: str, max_tool_turns: int = 10):
        """
        Generator that streams responses with support for multi-step agentic workflows.
        Increased max_tool_turns to 10 to support longer chains.
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
            yield f"\n\n🔧 Step {turn + 1}: Executing {len(tool_calls)} tool(s)...\n\n"

            for tc in tool_calls:
                yield f"🔹 Running: {tc.function.name}\n"
                try:
                    raw = self._execute_tool_sync(tc)
                    tool_call_id = str(uuid.uuid4())

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": (
                            f"The tool '{tc.function.name}' has finished executing.\n"
                            f"Raw output:\n{raw}\n\n"
                            "Now continue with the next step of the workflow or explain the result."
                        ),
                    })

                    # Show result preview
                    preview = raw[:200] + "..." if len(raw) > 200 else raw
                    yield f"   Result: {preview}\n"

                except Exception as e:
                    yield f"❌ Tool error: {e}\n"

            # Stream follow-up explanation or next step
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

        yield "\n\n⚠️ Reached maximum number of steps. Task may be incomplete."

    # def chat_stream(self, user_text: str, max_tool_turns: int = 4):
    #     """
    #     Generator that streams responses (converted from async to sync).
    #     This avoids the async context manager issue with st.write_stream.
    #     """
    #     self.messages.append({"role": "user", "content": user_text})
    #
    #     for _ in range(max_tool_turns):
    #         stream = ollama.chat(
    #             model=self.model,
    #             messages=self.messages,
    #             tools=self.available_tools,
    #             stream=True,
    #         )
    #
    #         assistant_text = ""
    #         tool_calls = []
    #
    #         # Stream tokens as they arrive
    #         for chunk in stream:
    #             if chunk.message.content:
    #                 assistant_text += chunk.message.content
    #                 yield chunk.message.content
    #             if chunk.message.tool_calls:
    #                 tool_calls.extend(chunk.message.tool_calls)
    #
    #         self.messages.append(
    #             {"role": "assistant", "content": assistant_text, "tool_calls": tool_calls or []}
    #         )
    #
    #         if not tool_calls:
    #             return
    #
    #         # Execute tools synchronously
    #         yield f"\n\n🔧 Executing {len(tool_calls)} tool(s)...\n\n"
    #
    #         for tc in tool_calls:
    #             yield f"🔹 Running: {tc.function.name}\n"
    #             try:
    #                 raw = self._execute_tool_sync(tc)
    #                 tool_call_id = str(uuid.uuid4())
    #
    #                 self.messages.append({
    #                     "role": "tool",
    #                     "tool_call_id": tool_call_id,
    #                     "content": (
    #                         f"The tool '{tc.function.name}' has finished executing.\n"
    #                         f"Raw output:\n{raw}\n\n"
    #                         "Now explain this result to the user in a clear, human-readable way."
    #                     ),
    #                 })
    #             except Exception as e:
    #                 yield f"❌ Tool error: {e}\n"
    #
    #         # Stream follow-up explanation
    #         stream2 = ollama.chat(
    #             model=self.model,
    #             messages=self.messages,
    #             tools=self.available_tools,
    #             stream=True,
    #         )
    #
    #         followup_text = ""
    #         for chunk in stream2:
    #             if chunk.message.content:
    #                 followup_text += chunk.message.content
    #                 yield chunk.message.content
    #
    #         self.messages.append({"role": "assistant", "content": followup_text})
    #
    #         if not getattr(chunk.message, "tool_calls", None):
    #             return
    #
    #     yield "\n\n⚠️ Couldn't complete within allowed tool-call steps."

