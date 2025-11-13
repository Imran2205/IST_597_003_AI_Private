P3 is built on P1, RAG assignemnts. It introduce tts-based output and stt-based input. it will take us closer to making our digital knowledge navigator. It relies less on P2 (fine-tuning and speech-based computer use) because once we have a solid information systme, P4 will use P2-based architecture to perform agentic interaction. 

In P3 we will create a knowledge based from pdf files. it will preprocess files and stored them in a persistent vector database, allows us to ask complex knowledge queiry, and perform file manipulation operation from natural language query. 
We will be using the same MCP server-client architecture; the server will host specialized tools using a library called RAG-Anything. 

common features include: 1) user will specify a data directory; 2) they ask the client to pre-process the document all at once, create a knowledge base, and store them in persistent storage. 3) next time, the user can simply reload the KB without doing the expensive pre-processing;
4) users can add new document from the UI 
5) users can ask complex queries and UI should be able to render the response
6) users can rename pdf files based on their title, regroup them into folders based on their content simillalrity


===========================
Project_P2: Finetuning models for Domain-Specific Tasks
  
Project Overview
Warning: This part of the project is hard. Please start early.

This is part 2 of our semester-long project: Intelligent Human Assistant.

In part 1, we aimed to make the command line interface more usable by allowing users to enter natural language queries. In this part, we will add a new capability—making computer use easier with Voice Control (on iOS) or Voice Access (on Android). Let's call it Voice User Interface (VUI) to be agnostic to iOS or Android. Like command lines, VUIs are very rigid—users must memorize commands, and any minor errors in command syntax are discarded by the VUI system.

Our goal is to clean up users' utterances in real-time using a fine-tuned small language model (e.g., Gemma 3 270M). It takes the raw user command and returns an output that the VUI expects at that time.

To make this system practical, we use two devices: a phone where the VUI is running, and a laptop where the LLM chatbot is running. For now, the chatbot takes user commands via text input, cleans them up, and reads them out loud so that the phone (nearby) listens to the command and acts upon it.

Background
Please watch this short tutorial on how to use VUI for basic interaction.



Refer to your OS-specific VUI documentation:

iOS: https://support.apple.com/guide/iphone/use-voice-control-iph2c21a3c88/ios Links to an external site.
Android: https://support.google.com/accessibility/android/answer/6151848?hl=en Links to an external site.
 

Architecture
Similar to Part 1, the system includes:

An MCP server hosting several tools
An MCP client hosting a locally-run LLM and offering a terminal-based text input
However, unlike Part 1, the MCP client now:

Runs through a graphical user interface based on Streamlit (see HW-RAG)
Has speech output capability (through the pyttsx3 library)
Additionally, the MCP server has a new tool that internally calls the smaller, fine-tuned LLM (let's call it Voice Command Converter or VCC). You need to fine-tune this model.

Users can type two types of commands: /echo [command] and /vc [command]. /echo commands are immediately read out by the interface.

Here is a demo of how /echo commands can be used:



 /vc commands will go through the entire MCP pipeline (UI -> MCP client -> MCP server -> VCC)

 

The system architecture is shown below:



 

Expected outcome:



Implementation Guide
Get yourself familiarize with VUI commands
Use the Gemma 3 (270M) for VCC
Use this Unsloth example notebook as your base: Gemma 3 (270M) — Unsloth Colab Example Links to an external site.
Run the notebook in Google Colab (Unsloth does not currently work on macOS). Use your PSU email to get premier tier GPU resource in Colab
Replace the example dataset with your dataset (instruction below). You can upload your dataset to Google Colab and use its path.
Modify the dataset mapper, tokenization, and training code as needed to handle your text pairs.
Train and evaluate your fine-tuned model.
Save the final model and tokenizer.
Test a few unseen voice commands and verify that the model outputs the expected structured format. You can also create a small test dataset with 100-150 samples and evaluate the model on that. In that case, you can either do string matching to see how accurate the model is or use the ROUGE score. Use this code snippet Download code snippet to compute the ROUGE score
Export the fine-tuned model (as GGUF format) to a local directory in Colab. Please download the saved model and the generated Modelfile immediately; otherwise, the file might be lost since Colab terminates the session if it is inactive for a while.
Put the GGUF file and the Modelfile in the same directory on your local computer, and run the following command from that directory to add the model in Ollama:
      ollama create gemma-3-270m-vc-finetuned -f ./Modelfile 

 

After you have a working, fine-tuned model (VCC) integrate it as a callable tool on your MCP server. Use the model 'gemma-3-270m-vc-finetuned' for implementing the tool.

In rag_server.py, add a new tool named, for example:
@mcp.tool()
async def correct_command(query: str) -> str:
"""
 Translate natural voice commands into standardized structured commands.
 Args:
      query: The user’s natural-language command.
 Returns:
       The translated voice command as a plain string.
 """
# your code here
Inside the tool:
Load your fine-tuned model using ollama.
Generate the translated command given the input prompt.
Return the translated command as a string.
In the UI side, route /echo commands and /vc commands properly (see the architecture diagram)
On the client side:
In rag_client.py, modify the system prompt to make sure the LLM understands which one is the voice command and which one is the previously implemented commands (e.g., terminal, my_files, arithmetic operations)
Implement a TTS component (e.g., using `pyttsx3`) that reads out the translated command in real time. Upon receiving the cleaned up command from the VCC,  make sure it only reads out the voice command.
 

Dataset Creation
You must create your own dataset for training. In fact, dataset creation is the main challenge to fine-tuning any model.

In your dataset, each entry should have the following five attributes. Feel free to use ChatGPT or other AI tools to get help with this step. Once this dataset is created, export it as a .jsonl file. A few entries from our dataset is shown below.
Include at least 300–500 examples of diverse natural language voice commands.
Cover synonyms, misspellings, or extra words users might say (e.g., “please,” “hey,” etc.).
Sample rows:

task	natural_command	input	expected_output	conversations
Convert the following natural language command to the correct voice control command format.	I was wondering if you could select 'tonight'	I was wondering if you could select 'tonight' | selection:	SELECT tonight	[{"content":"Convert the following natural language command to the correct voice control command format.","role":"system"},{"content":"I was wondering if you could select 'tonight' | selection: ","role":"user"},{"content":"SELECT tonight","role":"assistant"}]
Convert the following natural language command to the correct voice control command format.	alright umm redo this	alright umm redo this | selection:	REDO THAT	[{"content":"Convert the following natural language command to the correct voice control command format.","role":"system"},{"content":"alright umm redo this | selection: ","role":"user"},{"content":"REDO THAT","role":"assistant"}]
Convert the following natural language command to the correct voice control command format.	get text now	get text now | selection:	SELECT now	[{"content":"Convert the following natural language command to the correct voice control command format.","role":"system"},{"content":"get text now | selection: ","role":"user"},{"content":"SELECT now","role":"assistant"}]
Below are the lines from our dataset.jsonl file representing the above table. You can download it from here Download here:

image.png

 

Submission Guidelines
Submit a zip file containing:

All your code:
rag_server.py (with new tool)
rag_client.py (updated prompt + TTS)
ui.py
voice_commands.jsonl (the dataset)
Training notebook (modified Colab notebook)
A short demo video showing:
Model translation
TTS output
Real-world device control demonstration
A brief text file report.txt summarizing:
Dataset creation process
Model used and fine-tuning parameters
Example outputs
Any issues or potential improvements
============================
Project_P1: Towards Intelligent Assistance: A Natural Language Terminal

Learning Objective
This is the first part of our semester-long project towards creating an intelligent, multimodal assistant. In this part, we will be creating a command-line terminal where users can query in natural language. A command line interface is arguably the most poorly designed interface because the names of the commands are not intuitive and the number of parameters are somewhat arbitrary. As such, users must memorize the syntax of a command. It may work for expert users but not for regular computer users.

By completing this assignment, you will:

Gain hands-on experience with how to use local, large, language model to create LLM-powered app
How to create an agentic interaction with tool usage and MCP protocol
Learn how to wrap command-line operations as callable tools in a server.
Use of streaming model outputs and maintaining history to support deictic conversation.
Explore how to handle tool results in a conversational setting.
Project Description
You will implement an MCP-based system consisting of:

MCP Server (Terminal Tool)
Build a server that exposes terminal-related operations as MCP tools:
initiate_terminal: starts a persistent shell session. 
run_command: executes the given bash command within the terminal and returns its output.
terminate_terminal: safely closes the shell session.
Ensure the server maintains state (e.g., directory changes) across commands.
Handle process I/O properly using Python’s subprocess.
Example:
Client: cd ~/Desktop,    Tool: (changes directory inside the terminal session)
Client: ls,   Tool: returns list of files in ~/Desktop
MCP Client (Conversational Agent)
Build a client that connects to the MCP server.
Use Ollama as the underlying LLM (gpt-oss:20b or llama3.2:3b)
Maintain a loop for the conversation where the user enters either natural language or direct commands.
The model should:
Translate natural requests into commands (e.g., “What’s in this folder?” → ls -la).
Verify direct commands before sending to the tool.
Summarize terminal outputs into human-readable explanations.
Example:
User: What files are in this directory?
Model: decides which tool to use and the command → call_tool(tool_name="run_command", argument="ls -la")
Tool: returns terminal output
Model: explains → "There are 3 files: assignment.pdf, notes.txt, and data.csv."
Here is how the system should work:



Starter Code
You will be provided with starter scripts to help you. The two scripts below are included only as demonstrations of how an MCP server and client are built.

weather_server.py Download weather_server.py: This code implements an MCP server that provides tools to fetch and format weather alerts and forecasts from the U.S. National Weather Service (NWS) API.
weather_client.py Download weather_client.py: This code implements an MCP client that connects to a weather tool server, uses an Ollama model to interpret user queries, invokes the weather tools when needed, and returns natural-language responses about forecasts and alerts in an interactive chat loop.
Note: Their use case (weather information) is different from ours. These demos originate from the MCP website’s documentation Links to an external site.. They have been slightly modified to work with the Ollama model, whereas the original documentation used Claude.

You are also provided with the server script for our use case, which implements MCP tools for terminal use. The client script will be written by you as part of the assignment.

terminal_server.py Download terminal_server.py: This code implements an MCP server that provides tools to start, run commands in, and terminate a persistent terminal session.
References/Documentation
Whiteboard (system workflow): https://zoom.us/wb/doc/lJdR2538TNyf56f60rZkPw/p/239991448010752 Links to an external site.
MCP documentation for building the server: https://modelcontextprotocol.io/docs/develop/build-server Links to an external site.
MCP documentation for building the client: https://modelcontextprotocol.io/docs/develop/build-client Links to an external site.
Ollama documentation for using the tool: https://ollama.com/blog/tool-support Links to an external site.
Ollama documentation for streaming the model response: https://ollama.com/blog/streaming-tool Links to an external site.
Submission
A zip file containing:

Source code
A demo video/screencast of you using the system
Good luck, and enjoy building your intelligent conversational agent!

============
HW_RAG: A RAG-Based Personal Knowledge Assistant
  
Learning goals
Understand the retrieval stage in a simple RAG pipeline.

Extend an MCP-based server with a new tool.

Ingest a local text corpus into a FAISS-backed vector store.

Query the corpus end-to-end through the provided UI.

 

Starter kit
You are given three files we walked through in class:

rag_server.py — FastMCP server with FAISS vector store and a few predefined tools

rag_client.py — An MCP client that uses local models via Ollama, supports tool calling, and connects to the MCP server.

ui.py — Streamlit-based user interface for the client 

Assume the server already exposes query_knowledge(question: str) -> str that searches the shared vector store.

 

Your task
Create a subdirectory my_files/ next to rag_server.py. Put at least 5 nontrivial .txt files there (≥ 500 chars each).

Add a new MCP tool in rag_server.py that:

Reads all *.txt under ./my_files/ (recursively OK).

Splits each file into ~1000-character chunks.

Prefixes each chunk with its relative path.

Stores each chunk in the vector storage (to facilitate RAG). 

Returns a brief summary string (e.g., files ingested, chunks added, total docs).

You do not need to make any change in the ui.py
Feel free to make any change in the client (although we believe rag_client.py will work as-is)
Expected output (for inspiration):

 

Answer the following questions:
What problems does a RAG-based architecture solve compared to pure prompting on a frozen LLM
If the corpus changes over time, describe one approach to keep the index fresh without rebuilding everything from scratch.
 

Submission
A zip file containing:

Source code + text files (in your my_files folder) + answers to the questions in #3
A demo video/screencast of you using the system