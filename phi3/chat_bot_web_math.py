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
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.agents import AgentOutputParser  
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import AgentAction, AgentFinish
import requests
import json
from typing import Any, List, Mapping, Optional, Union
import logging
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Arithmetic operation functions
def add(input_str: str):
    a, b = map(float, input_str.split(','))
    """Add two numbers"""
    try:
        result = float(a) + float(b)
        return f"The result of {a} + {b} is {result}"
    except ValueError:
        return "Error: Please provide valid numbers for addition."

def subtract(input_str: str):
    a, b = map(float, input_str.split(','))
    try:
        result = float(a) - float(b)
        return f"The result of {a} - {b} is {result}"
    except ValueError:
        return "Error: Please provide valid numbers for subtraction."

def multiply(input_str: str):
    a, b = map(float, input_str.split(','))
    try:
        result = float(a) * float(b)
        return f"The result of {a} * {b} is {result}"
    except ValueError:
        return "Error: Please provide valid numbers for multiplication."

def divide(input_str: str):
    a, b = map(float, input_str.split(','))
    try:
        a, b = float(a), float(b)
        if b == 0:
            return "Error: Division by zero is not allowed."
        result = a / b
        return f"The result of {a} / {b} is {result}"
    except ValueError:
        return "Error: Please provide valid numbers for division."

class CustomLLM(LLM):
    api_url: str = "http://127.0.0.1:8899/v1/completions"

    def _call(
        self,
        prompt: str,
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

def simple_extract(content: str):
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

# Define the tools
tools = [
    Tool(
        name="Addition",
        func=add,
        description="Useful for adding two numbers together. Input should be two numbers separated by a comma."
    ),
    Tool(
        name="Subtraction",
        func=subtract,
        description="Useful for subtracting one number from another. Input should be two numbers separated by a comma."
    ),
    Tool(
        name="Multiplication",
        func=multiply,
        description="Useful for multiplying two numbers. Input should be two numbers separated by a comma."
    ),
    Tool(
        name="Division",
        func=divide,
        description="Useful for dividing one number by another. Input should be two numbers separated by a comma."
    ),
    Tool(
        name="Wikipedia_Scraper",
        func=scrape_wikipedia,
        description="Useful for scraping information from a Wikipedia page. Input should be a complete Wikipedia URL."
    ),
    Tool(
        name="Information_Query",
        func=query_vector_store,
        description="Useful for querying information from scraped Wikipedia pages. Input should be a question about the scraped content."
    )
]

# Set up the prompt template
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs):
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

# prompt_template = """Answer the following questions as best you can. You have access to the following tools:

# {tools}

# Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question

# For arithmetic operations, ALWAYS use the corresponding tools (Addition, Subtraction, Multiplication, Division) directly. Do not use Wikipedia_Scraper or Information_Query for arithmetic.
# Use Wikipedia_Scraper only for scraping web pages.
# Use Information_Query only for querying previously scraped information.

# Once you have answered all the questions in the input, respond with:
# Task Complete: Yes

# Begin!

# Question: {input}
# {agent_scratchpad}"""

prompt_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Important instructions:
1. Do not make up any questions and answers beyond what is asked.
2. Do not repeat the question in your Final Answer.
3. When you have finished answering ALL parts of the question, end your response with the marker [END_OF_RESPONSE].
4. Do not add any text, questions, or conversation after the [END_OF_RESPONSE] marker.

Begin!

Question: {input}
{agent_scratchpad}"""

prompt = CustomPromptTemplate(
    template=prompt_template,
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str):
        # Check for the end-of-response marker
        if "[END_OF_RESPONSE]" in llm_output:
            response = llm_output.split("[END_OF_RESPONSE]")[0].strip()
            
            # Extract only the Final Answer
            if "Final Answer:" in response:
                final_answer = response.split("Final Answer:")[-1].strip()
                return AgentFinish(
                    return_values={"output": final_answer},
                    log=llm_output,
                )
            else:
                return AgentFinish(
                    return_values={"output": response},
                    log=llm_output,
                )

        # Check if this is the final answer
        if "Final Answer:" in llm_output:
            final_answer = llm_output.split("Final Answer:")[-1].strip()
            if "[END_OF_RESPONSE]" in final_answer:
                final_answer = final_answer.split("[END_OF_RESPONSE]")[0].strip()
            return AgentFinish(
                return_values={"output": final_answer},
                log=llm_output,
            )

        # If it's not the final answer, parse the action
        pattern = r"Action: (.*?)\nAction Input: (.*?)(?=\n|$)"
        match = re.search(pattern, llm_output, re.DOTALL)

        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            
        action = match.group(1).strip()
        action_input = match.group(2).strip()
    
        logger.info(f'>>>> {action}: {action_input}')

        return AgentAction(tool=action, tool_input=action_input, log=llm_output)

        

output_parser = CustomOutputParser()

# Set up the agent
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, max_iterations=10)

def chat(message, history):
    try:
        response = agent_executor.run(message)
        # Ensure we're only returning the final answer
        if isinstance(response, dict) and "output" in response:
            return response["output"]
        elif isinstance(response, str):
            # If it's a string, it should already be the final answer, but let's make sure
            if "[END_OF_RESPONSE]" in response:
                response = response.split("[END_OF_RESPONSE]")[0].strip()
            return response
        else:
            return "I apologize, but I couldn't generate a proper response. Please try again."
    except Exception as e:
        logger.error(f"An error occurred during chat: {str(e)}")
        error_message = str(e)
        if "Could not parse LLM output" in error_message:
            return "I'm sorry, I couldn't process that request correctly. Could you please rephrase your question?"
        return f"I apologize, but I encountered an error. Please try again or rephrase your message."


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
with gr.Blocks(css=custom_css) as iface:
    with gr.Column():
        chatbot = gr.Chatbot(elem_id="chatbot-container")
        with gr.Row(elem_id="input-container"):
            msg = gr.Textbox(
                show_label=False,
                placeholder="Type your message here... (Use 'scrape:' for Wikipedia URLs or ask arithmetic questions)",
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

# Launch the interface
iface.launch()