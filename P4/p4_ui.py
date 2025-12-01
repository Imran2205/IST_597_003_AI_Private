import streamlit as st
import asyncio
from p4_rag_client import OllamaMCPClient
import pyttsx3, threading

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
        st.session_state.model_name = "gpt-oss:120b-cloud"


async def connect_to_server(server_url: str, model: str):
    """Connect to MCP server and initialize tools."""
    client = OllamaMCPClient(model=model, server_url=server_url)
    success, message = await client.initialize_tools()
    return client, success, message

# engine = pyttsx3.init()
# engine.setProperty('rate', 125)
# engine.say("")
# engine.runAndWait()

import threading


def speak_text(text_to_speak: str):
    def speak():
        """The actual work to be done in the thread."""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text_to_speak)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[TTS Error] Could not speak: {e}")

    tts_thread = threading.Thread(target=speak, daemon=True)
    tts_thread.start()


def main():
    st.set_page_config(page_title="MCP for Voice Command", page_icon="🤖", layout="wide")

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
            help="Name of the Ollama model to use (e.g., qwen2.5:14b, gptoss:20b)"  # Updated hint
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
        ### 💡 Multi-Step Voice Commands
        - **Simple**: "click button 5"
        - **Complex**: "go to my home"
        - **With prefix**: "/VC: navigate to settings"
        - The agent will break down complex tasks automatically
        - Watch tool execution in real-time
        """)

    # Main chat interface
    st.title("🤖 Agentic MCP Voice Command System")
    st.markdown("Multi-step UI automation with OCR-powered screen understanding")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your command (e.g., 'go to home', '/VC: show numbers')...",
                               disabled=not st.session_state.connected):
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
                if prompt.startswith('/echo'):
                    speak_text(prompt.replace('/echo', ''))
                    full_response = prompt.replace('/echo', '')
                elif prompt.lower().startswith('/vc') or prompt.lower().startswith('/cc'):
                    full_response = st.write_stream(
                        st.session_state.client.chat_stream(prompt)
                    )

                    # Handle voice command output and TTS
                    # if "##VC##" in full_response:
                    #     command_parts = full_response.split("##VC##")
                    #     if len(command_parts) >= 2:
                    #         voice_command = command_parts[1]
                    #         speak_text(voice_command)
                    #         st.info(f"🎤 Voice Command: {voice_command}")
                else:
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

# def main():
#     st.set_page_config(page_title="MCP for Voice Command", page_icon="🤖", layout="wide")
#
#     init_session_state()
#
#     # Sidebar
#     with st.sidebar:
#         st.title("⚙️ Settings")
#
#
#         server_url = st.text_input(
#             "MCP Server URL",
#             value=st.session_state.server_url,
#             help="URL of your MCP server"
#         )
#
#         model_name = st.text_input(
#             "Ollama Model",
#             value=st.session_state.model_name,
#             help="Name of the Ollama model to use"
#         )
#
#         if st.button("Connect", type="primary", use_container_width=True):
#             with st.spinner("Connecting to MCP server..."):
#                 client, success, message = asyncio.run(connect_to_server(server_url, model_name))
#                 if success:
#                     st.session_state.client = client
#                     st.session_state.connected = True
#                     st.session_state.server_url = server_url
#                     st.session_state.model_name = model_name
#                     st.success(message)
#                 else:
#                     st.error(message)
#
#         if st.session_state.connected:
#             st.success("✅ Connected")
#             if st.button("Clear Chat History", use_container_width=True):
#                 st.session_state.messages = []
#                 if st.session_state.client:
#                     st.session_state.client.messages = [
#                         {"role": "system", "content": st.session_state.client.system_prompt}
#                     ]
#                 st.rerun()
#         else:
#             st.warning("⚠️ Not connected")
#
#         st.divider()
#         st.markdown("""
#         ### 💡 Tips
#         - Connect to your MCP server first
#         - Ask questions or request tool usage
#         - Responses stream in real-time
#         - Tool calls are executed automatically
#         """)
#
#     # Main chat interface
#     st.title("🤖 MCP for Voice Command")
#     st.markdown("Chat with your AI assistant powered by Ollama and MCP tools")
#
#     # engine.say("stop listening ")
#     # engine.runAndWait()
#
#     # Display chat history
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#
#
#     # Chat input
#     if prompt := st.chat_input("Type your message here...", disabled=not st.session_state.connected):
#         if not st.session_state.connected:
#             st.error("Please connect to the MCP server first!")
#             return
#
#         # Add user message
#         st.session_state.messages.append({"role": "user", "content": prompt})
#
#         with st.chat_message("user"):
#             st.markdown(prompt)
#
#         # Stream assistant response
#         with st.chat_message("assistant"):
#             try:
#                 # Use the synchronous generator (no async context managers)
#
#                 # print(full_response)
#                 if prompt.startswith('/echo'):
#                     # engine.say(prompt.replace('/echo', ''))
#                     speak_text(prompt.replace('/echo', ''))
#
#                     full_response = prompt.replace('/echo', '')
#                 else:
#                     full_response = st.write_stream(
#                         st.session_state.client.chat_stream(prompt)
#                     )
#                     if "##VC##" in full_response:
#                         speak_text(full_response.split("##VC##")[1])
#
#                 st.session_state.messages.append({"role": "assistant", "content": full_response})
#
#                 # tts
#                 # get_tts().say(full_response)
#                 # get_tts().runAndWait()
#
#             except Exception as e:
#                 error_msg = f"❌ Error: {e}"
#                 st.error(error_msg)
#                 st.session_state.messages.append({"role": "assistant", "content": error_msg})
#
#
# if __name__ == "__main__":
#     main()
