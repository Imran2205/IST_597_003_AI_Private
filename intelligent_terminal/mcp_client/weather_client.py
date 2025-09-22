import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv
import sys
import ollama
import uuid

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self, model):
        # Initialize session and client objects
        self.model = model
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command="python3",  # change it to python if your environement supoorts it
            args=[server_script_path],
            env=None,
        )

        # 2. create the transport layer between the client and the server
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))

        # 3. create input and output from the transport layer
        self.std_output, self.std_input = stdio_transport

        #  4. based on the input and output, create a session
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.std_output, self.std_input))

        # 5. finally, initialize the session
        await self.session.initialize()

        # 6. query the list of tools avaialable in the server
        response = await self.session.list_tools()

        # 7. the descriptor of a tool expected by ollama
        tool_descriptor = {
            "type": "function",
            "function": {
                "name": None,
                "description": None,
                "parameters": None,
            },
        }

        # 8. iterate over avaiable tools
        self.available_tools = []
        for tool in response.tools:
            tool_descriptor = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            self.available_tools.append(tool_descriptor)

        for tool in self.available_tools:
            print(f"Tool name: {tool['function']['name']}")

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        # Initial Claude API call
        response = ollama.chat(
            model=self.model,
            messages=messages,
            tools=self.available_tools
        )

        # Process response and handle tool calls
        final_text = []

        # for msg in response:
        if response.message.content:
            final_text.append(response.message.content)
            messages.append({
                "role": "assistant",
                "content": response.message.content
            })

        if response.message.tool_calls:
            for tool in response.message.tool_calls:
                tool_name = tool.function.name
                tool_args = tool.function.arguments

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                # print(result.content[0].text)

                tool_call_id = str(uuid.uuid4())
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content":  result.content[0].text
                })

                # Get next response from Claude
                described_tool_response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    tools=self.available_tools
                )

                if described_tool_response.message.content:
                    final_text.append(described_tool_response.message.content)

        # print(final_text)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient(model="llama3.2:3b")
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())