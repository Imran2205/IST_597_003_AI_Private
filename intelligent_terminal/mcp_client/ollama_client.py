import asyncio
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import ollama


class OllamaMCPClient:
    def __init__(self, model="gpt-oss:20b"):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.model = model

    async def connect(self, server_script_path: str):
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None,
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        tools = await self.session.list_tools()
        print("Connected. Tools:", [t.name for t in tools.tools])

    async def chat_loop(self):
        print("\nOllama MCP Client Started!")
        print("Type a natural query or bash command (type 'quit' to exit).\n")

        result = await self.session.call_tool("initiate_terminal", {"cwd": None})
        print(result.content[0].text)

        while True:
            try:
                user_query = input("\n$ ").strip()
                if user_query.lower() in {"quit", "exit"}:
                    break

                planning_prompt = f"""
    You are an assistant helping with terminal operations.
    The user asked: {user_query}

    1. If the input is natural language, translate it into the correct bash command.
    2. If the input is already a command, verify and return it as-is.
    3. Return ONLY the bash command, nothing else.
    """
                planning_response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": planning_prompt}],
                )
                command_to_run = planning_response["message"]["content"].strip()

                print(f"[GPT-OSS decided command: {command_to_run}]")

                result = await self.session.call_tool("run_command", {"command": command_to_run})
                raw_output = result.content[0].text

                explanation_prompt = f"""
    The command executed was:
    {command_to_run}

    The raw terminal output was:
    {raw_output}

    Explain the result to the user in a clear, human-readable way.
    If the file or folder requested does not exist, clearly state that.
    If listing a directory, summarize how many items exist and their names.
    If showing file metadata, summarize size, date, and other key info.
    """
                explanation_response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": explanation_prompt}],
                )
                print("\n" + explanation_response["message"]["content"].strip())

            except Exception as e:
                print(f"\nError: {str(e)}")

        result = await self.session.call_tool("terminate_terminal")
        print(result.content[0].text)

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = OllamaMCPClient()
    try:
        await client.connect(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
