# Install dependencies:
pip3 install "mcp[cli]" mcp anthropic python-dotenv
pip3 install ollama


# To run
## First, run the client

```
cd path/to/the/client/folder
python3 ollama_client_agentic.py /path/to/mcp_server.py
```


```
python3 mcp_client/weather_client.py mcp_server/weather_server.py
```

For example, on my system:

```
cd /Users/smb/Documents/code/ist_597_fall25/IST_597_003_AI_Private/intelligent_terminal/mcp_client
python3 ollama_client_agentic.py ../mcp_server/mcp_tool_server.py
```
