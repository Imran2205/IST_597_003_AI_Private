import os
import json
import logging
import requests
from typing import Optional, Any, List
from langchain.llms.base import LLM

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://127.0.0.1:8811"

# Function to get or create API key
def get_or_create_api_key():
    api_key_file = "api_key.txt"
    
    if os.path.exists(api_key_file):
        with open(api_key_file, "r") as f:
            api_key = f.read().strip()
        logger.info("Loaded existing API key")
        return api_key
    else:
        response = requests.post(f"{BASE_URL}/v1/register")
        if response.status_code == 200:
            api_key = response.json()["api_key"]
            with open(api_key_file, "w") as f:
                f.write(api_key)
            logger.info("Registered new user and saved API key")
            return api_key
        else:
            logger.error(f"Failed to register. Status code: {response.status_code}")
            return None

class CustomLLM(LLM):
    api_url: str = f"{BASE_URL}/v1/completions"
    api_key: str = get_or_create_api_key()
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
    ) -> str:
        if not self.api_key:
            return "Error: Unable to obtain API key. Please check your connection and try again."
        
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }
        
        data = {
            "prompt": prompt,
            "max_new_tokens": 512,
            "temperature": 0.1,
            "top_p": 0.5,
            "n": 1,
            "stop": stop,
            "repetition_penalty": 1.0,
            "encoder_repetition_penalty": 1.0,
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()['choices'][0]['text']
            return result.strip()
        except requests.exceptions.RequestException as e:
            return f"Sorry, I encountered an error: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "custom"

def test_llm():
    # Initialize the custom LLM
    llm = CustomLLM()
    
    # Test cases
    test_prompts = [
        "Write a short story about a robot:",
        # "Explain how a computer works:",
        # "Write a poem about nature:",
        # "What is the meaning of life?",
        # "Write a Python function to calculate factorial:"
    ]
    
    print("\n=== Starting LLM Tests ===\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest Case {i}:")
        print(f"Prompt: {prompt}")
        print("\nResponse:")
        response = llm._call(prompt)
        print(response)
        print("\n" + "="*50)

if __name__ == "__main__":
    test_llm()
