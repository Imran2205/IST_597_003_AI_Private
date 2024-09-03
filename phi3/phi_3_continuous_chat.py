import gradio as gr
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
import requests
import json
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomLLM(LLM):
    api_url = "http://127.0.0.1:8899/v1/completions"

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
            "prompt": prompt,
            "max_tokens": 150,
            "temperature": 0.7,
            "top_p": 1.0,
            "n": 1,
            "stop": stop
        }
        try:
            logger.info(f"Sending prompt to API: {prompt}")
            response = requests.post(self.api_url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            result = response.json()['choices'][0]['text']
            logger.info(f"Received response from API: {result}")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"

    @property
    def _llm_type(self):
        return "custom"

llm = CustomLLM()

# Initialize the conversation chain with a window memory
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferWindowMemory(k=3, return_messages=True)
)

def extract_ai_response(response):
    ai_response = response.strip()
    ai_response = re.split(r'\n\s*(?:Human:|What|How|Why|When|Where|Who|Is|Are|Can|Could|Should|Would|Will).*?$', ai_response, flags=re.MULTILINE|re.IGNORECASE)[0]
    return ai_response.strip()

def chat(message, history):
    try:
        full_response = conversation.predict(input=message)
        logger.info(f"Raw response from conversation: {full_response}")

        ai_response = extract_ai_response(full_response)
        logger.info(f"Extracted AI response: {ai_response}")

        if not ai_response.strip():
            return "I apologize, but I couldn't generate a response. Please try asking your question again."

        conversation.memory.chat_memory.add_user_message(message)
        conversation.memory.chat_memory.add_ai_message(ai_response)

        return ai_response
    except Exception as e:
        logger.error(f"An error occurred during chat: {str(e)}")
        return f"I apologize, but I encountered an error. Please try again or rephrase your message."

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

with gr.Blocks(css=custom_css) as gr_iface:
    with gr.Column():
        chatbot = gr.Chatbot(elem_id="chatbot-container")
        with gr.Row(elem_id="input-container"):
            msg = gr.Textbox(
                show_label=False,
                placeholder="Type your message here...",
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

gr_iface.launch()
