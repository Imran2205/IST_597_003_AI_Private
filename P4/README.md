- Install pip install paddleocr
```shell
pip install paddlepaddle
pip install paddleocr
```
- run the mcp server using:
```shell
python p4_mcp_server.py
```
- Add the mcp server to Open-WebUI
To add an MCP server:

Open ⚙️ Admin Settings → External Tools.
Click + (Add Server).
Set Type to MCP (Streamable HTTP).

Enter your Server URL: http://127.0.0.1:3000/mcpLinks to an external site.. Click on the "Verify Connection" button next to it to confirm the connection.

Auth details to None

Enter ID to be 1 (or any number) and Name to be equal to the name of the MCP server (e.g., rag-anything-server for P3) 

Save. If prompted, restart Open WebUI.
