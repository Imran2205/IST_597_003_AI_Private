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
        self.system_prompt = (
            "You are an AI assistant that can use MCP tools exposed by the server.\n"
            "- Only call tools by their exact names from the provided tool list.\n"
            "- Do not invent tool names.\n"
            "- When you call a tool, provide valid arguments matching its schema.\n"
            "- After any tool result, explain the result clearly to the user.\n"
        )

    async def initialize_tools(self):
        """Initialize connection and fetch tools (one-time operation)."""
        try:
            async with streamablehttp_client(url=self.server_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    response = await session.list_tools()
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

    def chat_stream(self, user_text: str, max_tool_turns: int = 4):
        """
        Generator that streams responses (converted from async to sync).
        This avoids the async context manager issue with st.write_stream.
        """
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
            yield f"\n\n🔧 Executing {len(tool_calls)} tool(s)...\n\n"

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
                            "Now explain this result to the user in a clear, human-readable way."
                        ),
                    })
                except Exception as e:
                    yield f"❌ Tool error: {e}\n"

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
                    followup_text += chunk.message.content
                    yield chunk.message.content

            self.messages.append({"role": "assistant", "content": followup_text})

            if not getattr(chunk.message, "tool_calls", None):
                return

        yield "\n\n⚠️ Couldn't complete within allowed tool-call steps."


# =========================
# Streamlit UI
# =========================
def init_session_state():
    """Initialize session state variables."""
    if "client" not in st.session_state:
        st.session_state.client = None
    if "connected" not in st.session_state:
        st.session_state.connected = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "server_url" not in st.session_state:
        st.session_state.server_url = "http://127.0.0.1:3000/mcp"
    if "model_name" not in st.session_state:
        st.session_state.model_name = "llama3.2:3b"


async def connect_to_server(server_url: str, model: str):
    """Connect to MCP server and initialize tools."""
    client = OllamaMCPClient(model=model, server_url=server_url)
    success, message = await client.initialize_tools()
    return client, success, message


def main():
    st.set_page_config(page_title="MCP + Ollama Chat", page_icon="🤖", layout="wide")

    init_session_state()

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")

        server_url = st.text_input(
            "MCP Server URL",
            value=st.session_state.server_url,
            help="URL of your MCP server"
        )

        model_name = st.text_input(
            "Ollama Model",
            value=st.session_state.model_name,
            help="Name of the Ollama model to use"
        )

        if st.button("Connect", type="primary", use_container_width=True):
            with st.spinner("Connecting to MCP server..."):
                client, success, message = asyncio.run(connect_to_server(server_url, model_name))
                if success:
                    st.session_state.client = client
                    st.session_state.connected = True
                    st.session_state.server_url = server_url
                    st.session_state.model_name = model_name
                    st.success(message)
                else:
                    st.error(message)

        if st.session_state.connected:
            st.success("✅ Connected")
            if st.button("Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.client:
                    st.session_state.client.messages = [
                        {"role": "system", "content": st.session_state.client.system_prompt}
                    ]
                st.rerun()
        else:
            st.warning("⚠️ Not connected")

        st.divider()
        st.markdown("""
        ### 💡 Tips
        - Connect to your MCP server first
        - Ask questions or request tool usage
        - Responses stream in real-time
        - Tool calls are executed automatically
        """)

    # Main chat interface
    st.title("🤖 MCP + Ollama Chat")
    st.markdown("Chat with your AI assistant powered by Ollama and MCP tools")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here...", disabled=not st.session_state.connected):
        if not st.session_state.connected:
            st.error("Please connect to the MCP server first!")
            return

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Stream assistant response
        with st.chat_message("assistant"):
            try:
                # Use the synchronous generator (no async context managers)
                full_response = st.write_stream(
                    st.session_state.client.chat_stream(prompt)
                )

                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                error_msg = f"❌ Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
