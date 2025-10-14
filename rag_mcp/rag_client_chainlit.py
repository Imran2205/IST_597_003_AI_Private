import asyncio
import uuid
from contextlib import AsyncExitStack
import chainlit as cl
import ollama
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


class OllamaMCPClient:
    def __init__(self, model: str):
        self.model = model
        self.messages: list[dict] = []
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.available_tools: list[dict] = []
        self.system_prompt = (
            "You are an AI assistant that can use MCP tools exposed by the server.\n"
            "- Only call tools by their exact names from the provided tool list.\n"
            "- Do not invent tool names.\n"
            "- When you call a tool, provide valid arguments matching its schema.\n"
            "- After any tool result, explain the result clearly to the user.\n"
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

        response = await self.session.list_tools()
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
        self.messages = [{"role": "system", "content": self.system_prompt}]

    async def _execute_single_tool(self, tool_call) -> str:
        """Execute one tool call against MCP and return raw text result."""
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments or {}
        result = await self.session.call_tool(tool_name, tool_args)
        if result.content and len(result.content) > 0 and getattr(result.content[0], "text", None):
            return result.content[0].text
        return "Tool executed but returned no content."

    async def chat_with_streaming(self, user_text: str, msg: cl.Message, max_tool_turns: int = 4):
        """Handle a single user turn with streaming to Chainlit."""
        self.messages.append({"role": "user", "content": user_text})

        for _ in range(max_tool_turns):
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
                    await msg.stream_token(chunk.message.content)
                    assistant_text += chunk.message.content
                if chunk.message.tool_calls:
                    tool_calls.extend(chunk.message.tool_calls)

            self.messages.append({
                "role": "assistant",
                "content": assistant_text,
                "tool_calls": tool_calls or []
            })

            if not tool_calls:
                await msg.send()
                return

            # Execute tools
            for tc in tool_calls:
                raw = await self._execute_single_tool(tc)
                tool_call_id = str(uuid.uuid4())
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"Tool '{tc.function.name}' result:\n{raw}",
                })

            # Stream follow-up explanation
            stream2 = ollama.chat(
                model=self.model,
                messages=self.messages,
                tools=self.available_tools,
                stream=True,
            )

            followup_text = ""
            for chunk in stream2:
                if chunk.message.content:
                    await msg.stream_token(chunk.message.content)
                    followup_text += chunk.message.content

            self.messages.append({"role": "assistant", "content": followup_text})
            await msg.send()

            if not getattr(chunk.message, "tool_calls", None):
                return

    async def cleanup(self):
        await self.exit_stack.aclose()


async def stream_response(client, user_text):
    """Async generator for Streamlit"""
    client.messages.append({"role": "user", "content": user_text})

    stream = ollama.chat(
        model=client.model,
        messages=client.messages,
        tools=client.available_tools,
        stream=True,
    )

    for chunk in stream:
        if chunk.message.content:
            yield chunk.message.content


st.title("🤖 MCP + Ollama Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(stream_response(client, prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})