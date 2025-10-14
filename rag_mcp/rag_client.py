"""
MCP Client with Custom LLM and Gradio Interface
Connects to FastMCP server and provides a chat interface
"""

import asyncio
import sys
import json
import re
import uuid
from contextlib import AsyncExitStack
from typing import Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import requests
import gradio as gr
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPClient:
    """MCP Client that connects to FastMCP server and uses custom LLM API"""
    
    def __init__(self, llm_api_url: str, api_key: str):
        """
        Initialize the MCP client
        
        Args:
            llm_api_url: URL of the custom LLM API
            api_key: API key for the LLM
        """
        self.llm_api_url = llm_api_url
        self.api_key = api_key
        
        # MCP protocol-specific variables
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        
        # Chat history
        self.messages = []
        
        # Available tools from MCP server
        self.available_tools = []
        
        # LLM API headers
        self.llm_headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }
    
    async def connect(self, server_url: str):
        """
        Connect to the MCP server
        
        Args:
            server_url: URL of the MCP server (e.g., http://127.0.0.1:3000/mcp)
        """
        try:
            logger.info(f"Connecting to MCP server at {server_url}...")
            
            # Create the transport layer
            transport = await self.exit_stack.enter_async_context(
                streamablehttp_client(url=server_url)
            )
            
            # Create input and output from the transport layer
            self.transport_output, self.transport_input, _ = transport
            
            # Create a session
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.transport_output, self.transport_input)
            )
            
            # Initialize the session
            await self.session.initialize()
            logger.info("Successfully connected and initialized MCP session")
            
            # Query the list of tools available on the server
            response = await self.session.list_tools()
            
            # Store available tools with their schemas
            self.available_tools = []
            for tool in response.tools:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
                self.available_tools.append(tool_info)
                logger.info(f"Registered tool: {tool.name}")
            
            logger.info(f"Successfully connected to MCP server with {len(self.available_tools)} tools")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {str(e)}")
            return False
    
    def create_system_prompt(self) -> str:
        """Create system prompt with tool descriptions"""
        tool_descriptions = []
        for tool in self.available_tools:
            params_desc = []
            if "properties" in tool["parameters"]:
                for param_name, param_info in tool["parameters"]["properties"].items():
                    param_type = param_info.get("type", "unknown")
                    param_desc = param_info.get("description", "No description")
                    params_desc.append(f"  - {param_name} ({param_type}): {param_desc}")
            
            tool_desc = f"""
Tool: {tool['name']}
Description: {tool['description']}
Parameters:
{chr(10).join(params_desc) if params_desc else '  No parameters'}"""
            tool_descriptions.append(tool_desc)
        
        tools_text = "\n\n".join(tool_descriptions)
        
        return f"""You are a helpful AI assistant that can use tools to help answer questions.

Available Tools:
{tools_text}

To use a tool, respond in this EXACT format:
TOOL_CALL: tool_name
PARAMETERS: {{"param1": value1, "param2": value2}}

After receiving a tool result, you can either:
1. Make another tool call if needed
2. Provide a final answer starting with "FINAL_ANSWER:"

Important rules:
- For arithmetic operations, ALWAYS use the corresponding tool (add, subtract, multiply, divide)
- To scrape Wikipedia, use the scrape_wikipedia tool with the full URL
- To answer questions about scraped content, use the query_knowledge tool
- When you receive a TOOL_RESULT, you MUST explain it to the user in natural language
- Always start your final response with "FINAL_ANSWER:" when you're done using tools
- Be concise and clear in your responses
- If you can't help or don't have enough information, say so clearly
"""
    
    def call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Call the custom LLM API
        
        Args:
            prompt: Prompt to send to the LLM
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLM response text
        """
        data = {
            "prompt": prompt + "\nAnswer:",
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 1.0,
            "n": 1,
            "stop": ["\n\n", "Human:"]
        }
        
        try:
            response = requests.post(
                self.llm_api_url,
                headers=self.llm_headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()['choices'][0]['text']
            return result.strip()
        except Exception as e:
            logger.error(f"LLM API call failed: {str(e)}")
            return f"Error calling LLM: {str(e)}"
    
    def parse_tool_call(self, response: str) -> Optional[tuple]:
        """
        Parse tool call from LLM response
        
        Returns:
            Tuple of (tool_name, parameters) or None if no tool call found
        """
        if "TOOL_CALL:" not in response:
            return None
        
        try:
            # Extract tool name
            tool_match = re.search(r"TOOL_CALL:\s*([\w_]+)", response)
            if not tool_match:
                return None
            tool_name = tool_match.group(1).strip()
            
            # Extract parameters
            params_match = re.search(r"PARAMETERS:\s*({.*?})", response, re.DOTALL)
            if not params_match:
                # Check if tool requires no parameters
                return (tool_name, {})
            
            params_str = params_match.group(1).strip()
            # Handle single quotes and convert to valid JSON
            params_str = params_str.replace("'", '"')
            parameters = json.loads(params_str)
            
            logger.info(f"Parsed tool call: {tool_name} with params: {parameters}")
            return (tool_name, parameters)
        except Exception as e:
            logger.error(f"Error parsing tool call: {str(e)}")
            return None
    
    async def execute_tool(self, tool_name: str, parameters: dict) -> str:
        """
        Execute a tool via MCP
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters to pass to the tool
            
        Returns:
            Tool execution result
        """
        try:
            logger.info(f"Executing MCP tool: {tool_name} with params: {parameters}")
            result = await self.session.call_tool(tool_name, parameters)
            
            # Extract text from MCP result
            if result.content and len(result.content) > 0:
                return result.content[0].text
            else:
                return "Tool executed but returned no content"
                
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def process_message(self, user_input: str, max_iterations: int = 10) -> str:
        """
        Process a user message with potential tool calls
        
        Args:
            user_input: User's input message
            max_iterations: Maximum number of tool calling iterations
            
        Returns:
            Final response to the user
        """
        # Build prompt with system message and history
        system_prompt = self.create_system_prompt()
        
        messages = [system_prompt]
        
        # Add recent conversation history (last 3 exchanges)
        for msg in self.messages[-6:]:
            messages.append(msg)
        
        messages.append(f"\nUser: {user_input}\nAssistant:")
        full_prompt = "\n".join(messages)
        
        iteration = 0
        current_prompt = full_prompt
        conversation_log = []
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{max_iterations}")
            
            # Get LLM response
            response = self.call_llm(current_prompt, max_tokens=500)
            conversation_log.append(("assistant", response))
            
            # Check if this is a final answer
            if "FINAL_ANSWER:" in response:
                final_answer = response.split("FINAL_ANSWER:")[-1].strip()
                
                # Update conversation history
                self.messages.append(f"User: {user_input}")
                self.messages.append(f"Assistant: {final_answer}")
                
                return final_answer
            
            # Try to parse tool call
            tool_call = self.parse_tool_call(response)
            
            if tool_call is None:
                # No tool call found, treat as final answer
                self.messages.append(f"User: {user_input}")
                self.messages.append(f"Assistant: {response}")
                return response
            
            # Execute the tool
            tool_name, parameters = tool_call
            tool_result = await self.execute_tool(tool_name, parameters)
            conversation_log.append(("tool", f"{tool_name}: {tool_result}"))
            
            # Add tool result to prompt for next iteration
            current_prompt += f"\n{response}\n\nTOOL_RESULT: {tool_result}\n\nNow explain this result to the user in clear, natural language. Start with 'FINAL_ANSWER:' when done.\n\nAssistant:"
        
        # Max iterations reached
        return "I apologize, but I couldn't complete the task within the allowed steps. Please try rephrasing your question."
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
        logger.info("MCP client cleaned up")


# ============================================================================
# Gradio Interface
# ============================================================================

class GradioInterface:
    """Gradio interface for the MCP client"""
    
    def __init__(self, client: MCPClient):
        self.client = client
        self.loop = None
    
    def chat(self, message: str, history: list) -> str:
        """
        Process chat message
        
        Args:
            message: User message
            history: Chat history (not used in current implementation)
            
        Returns:
            Bot response
        """
        try:
            # Run async function in the event loop
            if self.loop is None:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
            
            response = self.loop.run_until_complete(
                self.client.process_message(message)
            )
            return response
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def clear_chat(self):
        """Clear chat history"""
        self.client.messages = []
        return None
    
    async def clear_knowledge_async(self):
        """Clear knowledge base via MCP tool"""
        try:
            result = await self.client.execute_tool("clear_knowledge_base", {})
            return [[None, result]]
        except Exception as e:
            error_msg = f"Error clearing knowledge base: {str(e)}"
            logger.error(error_msg)
            return [[None, error_msg]]
    
    def clear_knowledge(self):
        """Wrapper for clear_knowledge_async"""
        if self.loop is None:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        return self.loop.run_until_complete(self.clear_knowledge_async())
    
    def create_interface(self):
        """Create and return Gradio interface"""
        custom_css = """
        #chatbot-container {
            height: calc(100vh - 230px) !important;
            overflow-y: auto;
        }
        #input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 20px;
            background-color: white;
            border-top: 1px solid #ccc;
        }
        """
        
        with gr.Blocks(css=custom_css, title="MCP Agent Chat") as iface:
            gr.Markdown("""
            # 🤖 MCP Agent Chatbot
            
            This chatbot uses the Model Context Protocol (MCP) to interact with tools:
            - **Arithmetic**: Add, subtract, multiply, divide numbers
            - **Wikipedia**: Scrape and query Wikipedia pages  
            - **Knowledge Base**: Query stored information using FAISS vector search
            
            **Example queries:**
            - "What is 25 + 17?"
            - "Calculate 100 divided by 4"
            - "Scrape https://en.wikipedia.org/wiki/Artificial_intelligence"
            - "What is artificial intelligence?" (after scraping)
            - "Tell me about machine learning" (after scraping relevant content)
            """)
            
            with gr.Column():
                chatbot = gr.Chatbot(elem_id="chatbot-container", height=500)
                with gr.Row(elem_id="input-container"):
                    msg = gr.Textbox(
                        show_label=False,
                        placeholder="Type your message here...",
                        container=False,
                        scale=4
                    )
                    send = gr.Button("Send", scale=1)
                
                with gr.Row():
                    clear = gr.Button("Clear Chat")
                    clear_kb = gr.Button("Clear Knowledge Base")
            
            def user(user_message, history):
                return "", history + [[user_message, None]]
            
            def bot(history):
                user_message = history[-1][0]
                bot_message = self.chat(user_message, history[:-1])
                history[-1][1] = bot_message
                return history
            
            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot
            )
            send.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot
            )
            clear.click(self.clear_chat, None, chatbot, queue=False)
            clear_kb.click(self.clear_knowledge, None, chatbot, queue=False)
        
        return iface



# ============================================================================
# Main
# ============================================================================

async def main():
    """Main function to run the client"""
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python mcp_client.py <mcp_server_url>")
        print("Example: python mcp_client.py http://127.0.0.1:3000/mcp")
        sys.exit(1)
    
    mcp_server_url = sys.argv[1]
        
    
    # Initialize MCP client
    client = OllamaMCPClient(model="llama3.2:3b")
    
    client = MCPClient(
        llm_api_url="http://127.0.0.1:8899/v1/completions",
        api_key=api_key
    )
    
    # Connect to MCP server
    connected = await client.connect(mcp_server_url)
    if not connected:
        print("Failed to connect to MCP server. Make sure the server is running.")
        sys.exit(1)
    
    # Create Gradio interface
    gradio_interface = GradioInterface(client)
    iface = gradio_interface.create_interface()
    
    try:
        # Launch Gradio interface
        logger.info("Launching Gradio interface on http://127.0.0.1:7860")
        print("\n" + "="*60)
        print("MCP Agent Client Started!")
        print("="*60)
        print(f"Connected to MCP server: {mcp_server_url}")
        print(f"Available tools: {len(client.available_tools)}")
        print("Gradio interface: http://127.0.0.1:7860")
        print("="*60 + "\n")
        
        iface.launch(share=False, server_port=7860)
    finally:
        # Clean up
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())