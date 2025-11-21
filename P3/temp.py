import asyncio
import uuid
import os
from typing import AsyncGenerator
from pathlib import Path

import streamlit as st
import ollama
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


# =========================
# Ollama + MCP Client
# =========================
class OllamaMCPClient:
    def __init__(self, llm_model: str, embedding_model:str, vlm_model:str, server_url: str, env_vars:dict[str, str]):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.vlm_model = vlm_model
        
        self.server_url = server_url
        
        self.messages: list[dict] = []
        self.available_tools: list[dict] = []
        self.env_vars = env_vars

        # Load directory paths from environment
        self.data_dir = env_vars.get("DATA_DIR", "")
        self.output_dir = env_vars.get("OUTPUT_DIR", "")
        self.rag_storage_dir = env_vars.get("RAG_STORAGE_DIR", "")
        
        # Load API keys from environment
        self.openrouter_api_key = env_vars.get("OPENROUTER_API_KEY", "")
        self.ollama_api_key = env_vars.get("OLLAMA_API_KEY", "")

        # Ensure API keys are set in environment for server to access
        os.environ["OPENROUTER_API_KEY"] = self.openrouter_api_key
        os.environ["OLLAMA_API_KEY"] = self.ollama_api_key
        
        self.system_prompt = f"""
You are P3, an intelligent Knowledge Navigator assistant. You help users manage and query their document knowledge base using RAG-Anything.

**IMPORTANT - Configured Directory Paths**:
- Data Directory (files): {self.data_dir}
- Output Directory (Parsed docs): {self.output_dir}
- RAG Storage (Vector DB): {self.rag_storage_dir}

Always use these exact paths when calling tools.

**Your Capabilities**:

1. **RAG Operations**:
   - initialize_rag_system: Set up the RAG system
   - preprocess_documents: Process all files from {self.data_dir}
   - query_knowledge: Search the knowledge base for information
   - add_document: Add a new file to the knowledge base
   - rename_pdfs_by_title: Rename PDF files based on their titles
   - clear_rag_storage: Clear all storage
   - get_rag_status: Check comprehensive system status (use detailed=True for file list)

2. **Terminal Operations** (Full bash access - navigate freely!):
   - initiate_terminal: Start a bash terminal (optional cwd parameter)
   - run_command: Execute ANY bash command
   - terminate_terminal: Close the terminal (only when user explicitly asks)

**Terminal is FLEXIBLE** - You can navigate anywhere and run any bash command:
- Navigation: `cd /path/to/folder`, `cd ~`, `cd ..`, `pwd`
- List files: `ls`, `ls -la`, `find . -name "*.pdf"`
- Copy files: `cp /source/file.pdf {self.data_dir}/`
- Move files: `mv /source/*.pdf {self.data_dir}/`
- View files: `cat file.txt`, `head -20 file.txt`
- Search: `grep "pattern" file.txt`, `find /path -type f -name "*.pdf"`
- Download: `wget URL -P {self.data_dir}/` or `curl -o {self.data_dir}/file URL`
- Multiple commands: Use `&&` or `;` to chain commands

**Common File Operations Examples**:
- Copy PDFs from Downloads: `cp ~/Downloads/*.pdf {self.data_dir}/`
- Find and copy all PDFs: `find ~/Documents -name "*.pdf" -exec cp {{}} {self.data_dir}/ \;`
- Download from web: `wget https://example.com/paper.pdf -P {self.data_dir}/`
- List large files: `find {self.data_dir} -type f -size +10M -exec ls -lh {{}} \;`
- Count files: `ls {self.data_dir} | wc -l`

**Initialization**:
- When user says "init", "initialize", or "setup": Call BOTH initialize_rag_system AND initiate_terminal
- initialize_rag_system parameters:
  - parser: "mineru" (default, best for PDFs) or "docling" (better for Office docs)
  - enable_vision: true (default)
- initiate_terminal: Can specify starting directory with `cwd` parameter (defaults to {self.data_dir})

**Workflow**:
1. User says "init" → Initialize both RAG system and terminal
2. Navigate and gather files → Use run_command with bash commands
3. Copy files to data directory → `cp /source/*.pdf {self.data_dir}/`
4. Process documents → preprocess_documents or add_document
5. Query → query_knowledge

**When users ask to**:
- "init", "initialize", "setup" → Call initialize_rag_system AND initiate_terminal
- "go to /path" or "navigate to folder" → run_command with `cd /path && ls -la`
- "copy files from X to data" → run_command with `cp /source/*.pdf {self.data_dir}/`
- "show me files in X" → run_command with `ls -la /path`
- "find all PDFs in X" → run_command with `find /path -name "*.pdf"`
- "download file from URL" → run_command with `wget URL -P {self.data_dir}/`
- "what's in my documents" or "query" → query_knowledge (after init)
- "process documents" → preprocess_documents (after init)
- "status" → get_rag_status

**Pro Tips for Terminal Usage**:
- Terminal persists state - `cd` commands persist across multiple run_command calls
- Use `pwd` to check current directory
- Chain commands with `&&` for multi-step operations
- Use wildcards: `*.pdf`, `*.docx`, `**/*.txt`
- Check before copying: `ls /source/*.pdf` then `cp /source/*.pdf {self.data_dir}/`

**Supported File Types**:
- Documents: PDF, DOCX, TXT, MD
- Images: PNG, JPG, JPEG

**Parsers**:
- MinerU (default): Best for PDFs with complex layouts, tables, equations
- Docling: Better for Office documents (DOCX, PPTX, XLSX), HTML

Be conversational and helpful. When user says "init", initialize both systems immediately.
        """

    async def initialize_tools(self):
        """
        Initialize connection and fetch tools (one-time operation).
        """
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
            tool_names = [t["function"]["name"] for t in self.available_tools]
            
            # Include directory info in connection message
            connection_msg = (
                f"✅ Connected! {len(self.available_tools)} tools available: {', '.join(tool_names)}\n\n"
                f"📁 Configured directories:\n"
                f"  • Data: {self.data_dir}\n"
                f"  • Output: {self.output_dir}\n"
                f"  • Storage: {self.rag_storage_dir}\n\n"
                f"🔑 API Keys loaded:\n"
                f"  • OpenRouter: {'✓' if self.openrouter_api_key else '✗'}\n"
                f"  • Ollama Cloud: {'✓' if self.ollama_api_key else '✗'}\n\n"
                f"📄 Supported file types:\n"
                f"  • Documents: PDF, DOCX, TXT, MD\n"
                f"  • Images: PNG, JPG, JPEG\n\n"
                f"🔍 Available parsers:\n"
                f"  • MinerU (default) - Best for PDFs\n"
                f"  • Docling - Better for Office docs\n\n"
                f"💡 Type 'init' to initialize both RAG system and terminal, then start working with your documents!"
            )
            return True, connection_msg
        except Exception as e:
            return False, f"❌ Connection failed: {e}"

    def _execute_tool_sync(self, tool_call) -> str:
        """
        Execute tool synchronously by creating a new event loop.
        """

        async def _do_execute():
            async with streamablehttp_client(url=self.server_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments or {}
                    
                    # Inject directory paths into tool calls if not specified
                    if tool_name == "preprocess_documents" and "data_directory" not in tool_args:
                        tool_args["data_directory"] = self.data_dir
                    
                    if tool_name == "rename_pdfs_by_title" and "directory" not in tool_args:
                        tool_args["directory"] = self.data_dir
                    
                    if tool_name == "add_document" and "file_path" in tool_args:
                        # If file_path is just a filename, prepend data_dir
                        file_path = tool_args["file_path"]
                        if not os.path.isabs(file_path):
                            tool_args["file_path"] = os.path.join(self.data_dir, file_path)
                    
                    if tool_name == "initiate_terminal" and "cwd" not in tool_args:
                        # Default terminal to data directory
                        tool_args["cwd"] = self.data_dir
                    
                    # Inject environment variables and model names into initialize_rag_system
                    if tool_name == "initialize_rag_system":
                        # Pass all environment variables
                        tool_args["env_vars"] = self.env_vars
                        # Pass model names if not already specified
                        if "llm_model" not in tool_args:
                            tool_args["llm_model"] = self.llm_model
                        if "embedding_model" not in tool_args:
                            tool_args["embedding_model"] = self.embedding_model
                        if "vlm_model" not in tool_args:
                            tool_args["vlm_model"] = self.vlm_model
                        # Default to mineru parser if not specified
                        if "parser" not in tool_args:
                            tool_args["parser"] = "mineru"

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
        Generator that streams responses (converted from async to sync).
        This avoids the async context manager issue with st.write_stream.
        """
        self.messages.append({"role": "user", "content": user_text})

        for turn in range(max_tool_turns):
            # Get LLM response with tools
            stream = ollama.chat(
                model=self.llm_model,
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

            # Save assistant message to history
            self.messages.append(
                {"role": "assistant", "content": assistant_text, "tool_calls": tool_calls or []}
            )

            # If no tool calls, conversation is complete
            if not tool_calls:
                return

            # Execute all requested tools
            yield f"\n\n🔧 Executing {len(tool_calls)} tool(s)...\n\n"

            for tc in tool_calls:
                # Show what tool is being called
                yield f"🔹 Running: `{tc.function.name}`"
                if tc.function.arguments:
                    # Show arguments in a clean way
                    args_display = []
                    for k, v in tc.function.arguments.items():
                        # Truncate long values
                        v_str = str(v)
                        if len(v_str) > 50:
                            v_str = v_str[:47] + "..."
                        args_display.append(f"{k}={v_str}")
                    yield f" with {', '.join(args_display)}"
                yield "\n"
                
                try:
                    # Execute the tool
                    raw = self._execute_tool_sync(tc)
                    tool_call_id = str(uuid.uuid4())

                    # Add tool result to conversation
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": (
                            f"The tool '{tc.function.name}' has finished executing.\n"
                            f"Raw output:\n{raw}\n\n"
                            "Now explain this result to the user in a clear, human-readable way. "
                            "Summarize the key points and suggest next steps if applicable."
                        ),
                    })
                    yield f"✅ Tool completed\n\n"
                    
                except Exception as e:
                    error_msg = f"❌ Tool error: {str(e)}\n\n"
                    yield error_msg
                    
                    # Add error to conversation
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": str(uuid.uuid4()),
                        "content": f"Error executing '{tc.function.name}': {str(e)}"
                    })

            # Get LLM's explanation of the tool results
            yield "💬 "
            stream2 = ollama.chat(
                model=self.llm_model,
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

            # Save follow-up to history
            self.messages.append({
                "role": "assistant", 
                "content": followup_text,
                "tool_calls": followup_tool_calls or []
            })

            # If there are more tool calls, continue the loop
            if not followup_tool_calls:
                return

        # Max iterations reached
        yield "\n\n⚠️ Maximum tool call iterations reached. Operation may be incomplete."