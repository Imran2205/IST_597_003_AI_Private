import asyncio
import sys
import uuid
from contextlib import AsyncExitStack
from typing import Optional

import gradio as gr
import ollama

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


# =========================
# Ollama + MCP Client
# =========================
class OllamaMCPClient:
    def __init__(self, model: str):
        self.model = model

        # Chat history
        self.messages: list[dict] = []

        # MCP protocol bits
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()

        # Tools exposed by MCP (Ollama function-calling descriptor format)
        self.available_tools: list[dict] = []

        # Single system prompt with MCP-style instructions
        self.system_prompt = (
            "You are an AI assistant that can use MCP tools exposed by the server.\n"
            "- Only call tools by their exact names from the provided tool list.\n"
            "- Do not invent tool names.\n"
            "- When you call a tool, provide valid arguments matching its schema.\n"
            "- After any tool result, explain the result clearly to the user.\n"
            "- If the requested resource/path doesn't exist, state it plainly.\n"
            "- If a tool returns an error, summarize the error in human terms.\n"
        )

    async def connect(self, server_url: str):
        """Connect to MCP server over HTTP stream and fetch tool schemas."""
        transport = await self.exit_stack.enter_async_context(
            streamablehttp_client(url=server_url)
        )
        self.transport_output, self.transport_input, _ = transport

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.transport_output, self.transport_input)
        )
        await self.session.initialize()

        # Discover tools
        response = await self.session.list_tools()
        self.available_tools = []
        for tool in response.tools:
            # Ollama's function-calling descriptor
            self.available_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,  # JSON Schema
                    },
                }
            )

        # Initialize chat history with system message
        self.messages = [{"role": "system", "content": self.system_prompt}]

    async def _execute_single_tool(self, tool_call) -> str:
        """Execute one tool call against MCP and return raw text result."""
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments or {}
        result = await self.session.call_tool(tool_name, tool_args)
        if result.content and len(result.content) > 0 and getattr(result.content[0], "text", None):
            return result.content[0].text
        return "Tool executed but returned no content."

    async def chat_once(self, user_text: str, max_tool_turns: int = 4) -> str:
        """
        Handle a single user turn:
        - Send message to model (with tool schemas)
        - Execute any requested tools
        - Feed results back for explanation
        - Repeat (bounded) if the model chains more tool calls
        """
        self.messages.append({"role": "user", "content": user_text})

        # Iteratively allow tool usage then explain results
        for _ in range(max_tool_turns):
            # Ask model, allowing tool calls
            stream = ollama.chat(
                model=self.model,
                messages=self.messages,
                tools=self.available_tools,
                stream=True,
            )

            assistant_text_chunks = []
            tool_calls = []

            # Collect streamed text and tool calls (if any)
            for chunk in stream:
                if chunk.message.content:
                    assistant_text_chunks.append(chunk.message.content)
                if chunk.message.tool_calls:
                    tool_calls.extend(chunk.message.tool_calls)

            assistant_text = "".join(assistant_text_chunks).strip()
            self.messages.append(
                {"role": "assistant", "content": assistant_text, "tool_calls": tool_calls or []}
            )

            # If no tools were requested, we are done
            if not tool_calls:
                return assistant_text if assistant_text else "(no response)"

            # Execute each tool, then ask the model to explain results
            for tc in tool_calls:
                raw = await self._execute_single_tool(tc)
                tool_call_id = str(uuid.uuid4())

                # Provide structured tool result to the model
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": (
                            f"The tool '{tc.function.name}' has finished executing.\n"
                            f"Raw output:\n{raw}\n\n"
                            "Now explain this result to the user in a clear, human-readable way. "
                            "If an error occurred, summarize it plainly."
                        ),
                    }
                )

            # Ask the model to produce the human explanation (no extra tools necessary here,
            # but we allow them in case the model wants to chain)
            stream2 = ollama.chat(
                model=self.model,
                messages=self.messages,
                tools=self.available_tools,
                stream=True,
            )

            assistant_followup_chunks = []
            for chunk in stream2:
                if chunk.message.content:
                    assistant_followup_chunks.append(chunk.message.content)

            followup = "".join(assistant_followup_chunks).strip()
            self.messages.append({"role": "assistant", "content": followup})

            # If the follow-up contains no new tool calls (most cases), return it
            # Otherwise, the loop will continue and allow another round
            if followup:
                # Quick peek if last streamed chunk asked for more tools (rare)
                # If Ollama included tool_calls, they'd be attached to the assistant message.
                if not getattr(chunk.message, "tool_calls", None):
                    return followup

        return "I couldn't complete this within the allowed tool-call steps. Try refining your request."

    async def cleanup(self):
        await self.exit_stack.aclose()


# =========================
# Gradio UI
# =========================
class GradioOllamaMCPApp:
    def __init__(self, client: OllamaMCPClient):
        self.client = client

    def build_interface(self):
        with gr.Blocks(title="MCP + Ollama Chat") as demo:
            gr.Markdown(
                """
                # 🤖 MCP + Ollama Chat
                
                This chat uses an **Ollama** model and calls **MCP server tools** when needed.
                - Tools are fetched from your MCP server automatically.
                - The model decides when to call a tool and explains results.

                **Tip:** Ask it to run a terminal command (if your MCP server exposes terminal tools),
                list files, read content, scrape pages, etc. (depending on your server).
                """
            )

            with gr.Row():
                mcp_url = gr.Textbox(
                    label="MCP Server URL",
                    value="http://127.0.0.1:3000/mcp",
                    interactive=True,
                )
                model_name = gr.Textbox(
                    label="Ollama Model",
                    value="llama3.2:3b",
                    interactive=True,
                )
                connect_btn = gr.Button("Connect")

            status = gr.Markdown("**Status:** Not connected.")
            chat = gr.Chatbot(height=500)
            msg = gr.Textbox(placeholder="Type your message...", show_label=False)
            send = gr.Button("Send")
            clear = gr.Button("Clear")

            # Handlers
            async def do_connect(url, model):
                try:
                    self.client.model = model.strip()
                    await self.client.connect(url.strip())
                    return f"**Status:** Connected. Tools available: {len(self.client.available_tools)}"
                except Exception as e:
                    return f"**Status:** Connection failed: {e}"

            async def send_msg(history, text):
                if not text.strip():
                    return history, ""
                history = history + [[text, None]]
                try:
                    reply = await self.client.chat_once(text.strip())
                except Exception as e:
                    reply = f"Error: {e}"
                history[-1][1] = reply
                return history, ""

            def do_clear():
                # Reset only visible chat; keep backend history if desired
                # If you want to reset the model history too, uncomment the next line:
                # self.client.messages = [{"role": "system", "content": self.client.system_prompt}]
                return []

            connect_btn.click(
                do_connect,
                inputs=[mcp_url, model_name],
                outputs=[status],
            )

            send.click(
                send_msg,
                inputs=[chat, msg],
                outputs=[chat, msg],
            )
            msg.submit(
                send_msg,
                inputs=[chat, msg],
                outputs=[chat, msg],
            )
            clear.click(do_clear, None, [chat])

        return demo


# =========================
# Main
# =========================
async def _amain():
    if len(sys.argv) < 2:
        print("Usage: python app.py http://127.0.0.1:3000/mcp [ollama_model]")
        print("If you don't pass args, you can still connect from the UI.")
    # Create client with optional CLI model
    model = sys.argv[2] if len(sys.argv) >= 3 else "llama3.2:3b"
    client = OllamaMCPClient(model=model)

    # Optional: connect on startup if URL provided
    if len(sys.argv) >= 2:
        try:
            await client.connect(sys.argv[1])
            print("Connected at startup. Tools:", len(client.available_tools))
        except Exception as e:
            print("Startup connect failed:", e)

    app = GradioOllamaMCPApp(client)
    demo = app.build_interface()
    # Launch Gradio (blocking)
    demo.queue().launch(server_port=7860, share=False)


if __name__ == "__main__":
    asyncio.run(_amain())
