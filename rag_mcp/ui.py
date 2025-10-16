import streamlit as st
import asyncio
from rag_client import OllamaMCPClient
import pyttsx3

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

engine = pyttsx3.init()
engine.setProperty('rate', 125)
engine.say("")
engine.runAndWait()


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
    st.title("🤖 MCP for Voice Command")
    st.markdown("Chat with your AI assistant powered by Ollama and MCP tools")

    # engine.say("stop listening ")
    # engine.runAndWait()

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

                print(full_response)
                if "##VC##" in full_response:
                    engine.say(full_response.split("##VC##")[1])

                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # tts
                # get_tts().say(full_response)
                # get_tts().runAndWait()

            except Exception as e:
                error_msg = f"❌ Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
