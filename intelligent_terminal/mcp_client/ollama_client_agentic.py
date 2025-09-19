import asyncio
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import ollama
import uuid


class OllamaMCPClient:
    def __init__(self, model="gpt-oss:20b"):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.model = model
        self.messages = []  # chat history

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

        # List tools from server
        response = await self.session.list_tools()
        self.available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in response.tools
        ]
        # print(self.available_tools)
        print("Connected. Tools:", [t["function"]["name"] for t in self.available_tools])

    async def chat_loop(self):
        print("\nOllama Agentic MCP Client Started!")
        print("Type a natural query or bash command (type 'quit' to exit).\n")

        self.messages.append({
            "role": "system",
            "content": "You are an AI assistant that can use terminal tools. You can access file system using the terminal."
                       "You can use the tools to execute bash commands. Please decide if you should use tools based on users request."
                       "Use the tools when necessary. Maintain context from prior conversation."
                       "- When you want to run a tool, ALWAYS include its exact name."
                       "- The available tool names from tool description provided."
                       "- Do not leave the function name blank."
                       "- Do not invent new tool names."
                       "- Whenever you receive a tool result, you must always explain it back to the user in natural language."
                       "- After every tool result, you must always respond to the user with a clear explanation."
                       "- When you get the response from the tool:"
                            "- Explain the result to the user in a clear, human-readable way."
                            "- If the file or folder requested does not exist, clearly state that."
                            "- If listing a directory, summarize how many items exist and their names."
                            "- If showing file metadata, summarize size, date, and other key info."
        }) # are: 'initiate_terminal', 'run_command', 'terminate_terminal'.

        result = await self.session.call_tool("initiate_terminal", {"cwd": ""})
        print(result.content[0].text)

        while True:
            try:
                user_query = input("\n$ ").strip()
                if user_query.lower() in {"quit", "exit"}:
                    break

                self.messages.append({"role": "user", "content": user_query})

                stream = ollama.chat(
                    model=self.model,
                    messages=self.messages,
                    tools=self.available_tools,
                    stream=True,
                )

                assistant_content = []
                tool_calls = []

                # print("\n#####MODEL THINKING#####")
                for chunk in stream:
                    if chunk.message.content:
                        print(chunk.message.content, end='', flush=True)
                        assistant_content.append(chunk.message.content)

                    if chunk.message.tool_calls:
                        tool_calls.extend(chunk.message.tool_calls)
                # print("\n#####END MODEL THINKING#####")

                assistant_msg = {
                    "role": "assistant",
                    "content": "".join(assistant_content),
                    "tool_calls": tool_calls,
                }

                self.messages.append(assistant_msg)

                # print(assistant_msg)

                if assistant_msg["content"] and not assistant_msg.get("tool_calls"):
                    # print("\n" + assistant_msg["content"])
                    continue

                if "tool_calls" in assistant_msg:
                    for tool_call in assistant_msg["tool_calls"]:
                        tool_name = tool_call.function.name
                        tool_args = tool_call.function.arguments

                        print(f"Model requested tool: {tool_name} {tool_args}")

                        tool_result = await self.session.call_tool(tool_name, tool_args)

                        # self.messages.append({
                        #     "role": "tool",
                        #     "tool_call_name": tool_call.function.name,
                        #     "content": f"Tool '{tool_call.function.name}' executed. Here is the raw terminal output:\n{tool_result.content[0].text}\n\nPlease explain this result clearly to the user."
                        # })

                        print(tool_call)
                        tool_call_id = str(uuid.uuid4())

                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": (
                                f"The tool '{tool_call.function.name}' has finished executing.\n"
                                f"Raw output:\n{tool_result.content[0].text}\n\n"
                                "Now explain this result to the user in a clear, human-readable way."
                            )
                        })

                        # print(tool_result.content[0].text[:20])

                        stream = ollama.chat(
                            model=self.model,
                            messages=self.messages,
                            tools=self.available_tools,
                            stream=True,
                        )

                        assistant_content = []
                        tool_calls = []

                        # print("\n#####MODEL FOLLOW-UP THINKING#####")
                        for chunk in stream:
                            if chunk.message.content:
                                print(chunk.message.content, end='', flush=True)
                                assistant_content.append(chunk.message.content)

                            if chunk.message.tool_calls:
                                tool_calls.extend(chunk.message.tool_calls)

                        print()
                        # print("\n#####END MODEL FOLLOW-UP THINKING#####")

                        followup_msg = {
                            "role": "assistant",
                            "content": "".join(assistant_content),
                        }
                        if tool_calls:
                            followup_msg["tool_calls"] = tool_calls

                        self.messages.append(followup_msg)

                        # if followup_msg["content"]:
                        #     print("\n" + followup_msg["content"])

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
