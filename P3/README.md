P3 is built on P1, RAG assignemnts. It introduce tts-based output and stt-based input. It will take us closer to making our digital knowledge navigator. It relies less on P2 (fine-tuning and speech-based computer use) because once we have a solid information systme, P4 will use P2-based architecture to perform agentic interaction. 

In P3 we will create a knowledge based from pdf files. it will preprocess files and stored them in a persistent vector database, allows us to ask complex knowledge queiry, and perform file manipulation operation from natural language query. 
We will be using the same MCP server-client architecture; the server will host specialized tools using a library called RAG-Anything. 

common features include: 1) user will specify a data directory; 2) they ask the client to pre-process the document all at once, create a knowledge base, and store them in persistent storage. 3) next time, the user can simply reload the KB without doing the expensive pre-processing;
4) users can add new document from the UI 
5) users can ask complex queries and UI should be able to render the response
6) users can rename pdf files based on their title, regroup them into folders based on their content simillalrity