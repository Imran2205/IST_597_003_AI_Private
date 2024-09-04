import gradio as gr
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
import requests
import json
import logging
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomLLM(LLM):
    api_url: str = "http://127.0.0.1:8899/v1/completions"

    def _call(
        self,
        prompt,
        stop = None,
        run_manager = None,
    ):
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "prompt": prompt + "\nAnswer:",
            "max_tokens": 150,
            "temperature": 0.7,
            "top_p": 1.0,
            "n": 1,
            "stop": stop or ["Human:", "\n\n"]
        }
        try:
            logger.info(f"Sending prompt to API: {prompt}")
            response = requests.post(self.api_url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            result = response.json()['choices'][0]['text']
            logger.info(f"Received response from API: {result}")
            return result.strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"

    @property
    def _llm_type(self):
        return "custom"

# Initialize the custom LLM
llm = CustomLLM()

# Initialize the conversation chain with a window memory
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferWindowMemory(k=3, return_messages=True)
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

# Initialize FAISS vector store
vector_store = None

def simple_extract(content):
    prompt = f"""
    Summarize the following Wikipedia content in a few sentences:

    {content[:1000]}  # Limited to 1000 characters

    Summary:
    """
    
    response = llm(prompt)
    logger.info(f"Extraction response: {response}")
    return response

def scrape_wikipedia(url):
    global vector_store
    try:
        # Validate URL
        result = urlparse(url)
        if not all([result.scheme, result.netloc]) or "wikipedia.org" not in result.netloc:
            return "Invalid URL. Please provide a complete Wikipedia URL."

        # Load web content
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the main content div
        content_div = soup.find('div', {'id': 'mw-content-text'})

        # Extract title
        title = soup.find('h1', {'id': 'firstHeading'}).text

        # Extract content
        content = []
        if content_div:
            for elem in content_div.find_all(['p', 'h2', 'h3']):
                if elem.name == 'p':
                    content.append(elem.text)
                elif elem.name in ['h2', 'h3']:
                    content.append(f"\n\n{elem.text}\n")

        full_content = f"{title}\n\n{''.join(content)}"
        logger.info(f"Scraped content (first 1000 chars): {full_content[:1000]}")

        # Extract content with simple function
        extracted_content = simple_extract(full_content)
        logger.info(f"Extracted content: {extracted_content}")

        # Create or update the vector store
        if vector_store is None:
            vector_store = FAISS.from_texts([extracted_content], embeddings)
        else:
            vector_store.add_texts([extracted_content])
        
        return f"Successfully scraped and extracted information from: {url}"
    except Exception as e:
        logger.error(f"Error scraping Wikipedia: {str(e)}")
        return f"Error scraping Wikipedia: {str(e)}"

def query_vector_store(query):
    if vector_store is None:
        return "No information has been scraped yet. Please provide a Wikipedia URL to scrape first."
    
    try:
        docs = vector_store.similarity_search(query, k=1)
        logger.info(f"Retrieved {len(docs)} documents from vector store")
        for i, doc in enumerate(docs):
            logger.info(f"Document {i + 1} content: {doc.page_content[:100]}...")  # Log first 100 chars of each document
        
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        
        logger.info(f"Generated response: {response}")
        
        if not response.strip():
            return "I apologize, but I couldn't generate a response based on the scraped information. Please try rephrasing your question."
        
        return response
    except Exception as e:
        logger.error(f"Error querying vector store: {str(e)}")
        return f"Error querying information: {str(e)}"

def chat(message, history):
    try:
        if message.lower().startswith("scrape:"):
            url = message[7:].strip()
            return scrape_wikipedia(url)
        elif vector_store is not None:
            return query_vector_store(message)
        else:
            full_response = conversation.predict(input=message)
            if not full_response.strip():
                return "I apologize, but I couldn't generate a response. Please try asking your question again."
            conversation.memory.chat_memory.add_user_message(message)
            conversation.memory.chat_memory.add_ai_message(full_response)
            return full_response
    except Exception as e:
        logger.error(f"An error occurred during chat: {str(e)}")
        return f"I apologize, but I encountered an error: {str(e)}. Please try again or rephrase your message."

# Custom CSS for full height
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

# Create the Gradio interface
with gr.Blocks(css=custom_css) as gr_iface:
    with gr.Column():
        chatbot = gr.Chatbot(elem_id="chatbot-container")
        with gr.Row(elem_id="input-container"):
            msg = gr.Textbox(
                show_label=False,
                placeholder="Type your message here... (Start with 'scrape:' to add a Wikipedia URL)",
                container=False
            )
            send = gr.Button("Send")
        clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        user_message = history[-1][0]
        bot_message = chat(user_message, history[:-1])
        history[-1][1] = bot_message
        return history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    send.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the Gradio interface
gr_iface.launch()
