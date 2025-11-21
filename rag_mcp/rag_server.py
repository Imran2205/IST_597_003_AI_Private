"""
MCP Server with FAISS Vector Store and Tools
Uses FastMCP for easy server setup with arithmetic and Wikipedia tools
"""
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Any
import httpx
import numpy as np
import torch
import os

vc_selection = ""

# Make absolutely sure we won't use MPS even if available
if hasattr(torch.backends, "mps"):
    # hard-disable MPS so libs don't probe it
    torch.backends.mps.is_available = lambda: False  # type: ignore
    torch.backends.mps.is_built = lambda: False

from pathlib import Path
import base64
import subprocess
import tempfile
from PIL import Image
import requests
import json

# Now import sentence_transformers and other torch-dependent libraries
import faiss
# faiss.omp_set_num_threads(1)
# torch.set_num_threads(1)

from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import logging
import asyncio
from mcp.server.fastmcp import FastMCP
from ollama import Client


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("llm-agent-tools", port=3000)


# Configure Ollama host (same default as your server)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
_ollama_client = Client(host=OLLAMA_HOST)

# Global vector store instance
class VectorStoreManager:
    """Manages FAISS vector store for the server"""
    
    def __init__(self):
        self.encoder = None
        self.dimension = None
        self.index = None
        self.documents = []
        self.initialized = False
    
    def initialize(self):
        """Lazy initialization of the embedding model"""
        if not self.initialized:
            logger.info("Initializing embedding model...")
            self.encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            self.dimension = self.encoder.get_sentence_embedding_dimension()
            logger.info(f"Embedding model initialized with dimension: {self.dimension}")
            self.initialized = True
    
    def add_texts(self, texts: list[str]):
        """Add texts to the vector store"""
        self.initialize()

        embeddings = self.encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
        embeddings = np.ascontiguousarray(embeddings.astype('float32'))

        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.index.add(embeddings)
        self.documents.extend(texts)
        logger.info(f"Added {len(texts)} documents. Total: {len(self.documents)}")
    
    def similarity_search(self, query: str, k: int = 2) -> list[str]:
        """Search for similar documents"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        self.initialize()
        
        query_embedding = self.encoder.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding).astype('float32')
        
        k = min(k, len(self.documents))
        distances, indices = self.index.search(query_embedding, k)
        
        results = [self.documents[idx] for idx in indices[0]]
        logger.info(f"Found {len(results)} similar documents")
        return results
    
    def clear(self):
        """Clear the vector store"""
        self.index = None
        self.documents = []
        logger.info("Vector store cleared")

# Global vector store instance
vector_store = VectorStoreManager()

#region Arithmetic Tools for quick testing
@mcp.tool()
async def add(a: float, b: float) -> str:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        String containing the sum result
    """
    try:
        result = float(a) + float(b)
        print("Calling add tool")
        return f"The result of {a} + {b} is {result}"
    except (ValueError, TypeError) as e:
        return f"Error: Invalid numbers provided for addition. {str(e)}"


@mcp.tool()
async def subtract(a: float, b: float) -> str:
    """Subtract b from a.
    
    Args:
        a: Number to subtract from
        b: Number to subtract
    
    Returns:
        String containing the difference result
    """
    try:
        result = float(a) - float(b)
        print("Calling subtract tool")
        return f"The result of {a} - {b} is {result}"
    except (ValueError, TypeError) as e:
        return f"Error: Invalid numbers provided for subtraction. {str(e)}"


@mcp.tool()
async def multiply(a: float, b: float) -> str:
    """Multiply two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        String containing the product result
    """
    try:
        result = float(a) * float(b)
        print("Calling multiply tool")
        return f"The result of {a} * {b} is {result}"
    except (ValueError, TypeError) as e:
        return f"Error: Invalid numbers provided for multiplication. {str(e)}"


@mcp.tool()
async def divide(a: float, b: float) -> str:
    """Divide a by b.
    
    Args:
        a: Dividend (number to be divided)
        b: Divisor (number to divide by)
    
    Returns:
        String containing the quotient result or error if dividing by zero
    """
    try:
        print("Calling divide tool")
        a, b = float(a), float(b)
        if b == 0:
            return "Error: Division by zero is not allowed."
        result = a / b
        return f"The result of {a} / {b} is {result}"
    except (ValueError, TypeError) as e:
        return f"Error: Invalid numbers provided for division. {str(e)}"

#endregion arithmatic tools

#region Wikipedia Tool
@mcp.tool()
async def scrape_wikipedia(url: str) -> str:
    """Scrape and store information from a Wikipedia page.
    
    This tool will:
    1. Fetch and parse Wikipedia content
    2. Extract the title and main text
    3. Generate embeddings and store in FAISS vector database
    4. Make the content available for querying
    
    Args:
        url: Complete Wikipedia URL to scrape (e.g., https://en.wikipedia.org/wiki/Python)
    
    Returns:
        Success message with title or error message
    """
    try:
        # Validate URL
        print("Calling scrape tool")
        result = urlparse(url)
        if not all([result.scheme, result.netloc]) or "wikipedia.org" not in result.netloc:
            return "Error: Invalid URL. Please provide a complete Wikipedia URL (e.g., https://en.wikipedia.org/wiki/Topic)"

        logger.info(f"Scraping Wikipedia URL: {url}")

        # Fetch web content
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the main content div
        content_div = soup.find('div', {'id': 'mw-content-text'})

        # Extract title
        title_elem = soup.find('h1', {'id': 'firstHeading'})
        title = title_elem.text if title_elem else "Unknown Title"

        # Extract content
        content = []
        if content_div:
            for elem in content_div.find_all(['p', 'h2', 'h3']):
                if elem.name == 'p':
                    text = elem.get_text().strip()
                    if text and len(text) > 20:  # Filter out very short paragraphs
                        content.append(text)
                elif elem.name in ['h2', 'h3']:
                    content.append(f"\n\n{elem.get_text()}\n")

        full_content = f"{title}\n\n{''.join(content)}"
        logger.info(f"Scraped {len(full_content)} characters from {title}")

        # Create summary (first 5000 characters as context)
        summary = full_content[:5000]
        
        # Store in vector database
        # Break content into chunks for better retrieval
        chunk_size = 1000
        chunks = []
        
        # Add title and summary as first chunk
        chunks.append(f"Title: {title}\n\nSummary: {summary}")
        
        # Add content chunks
        for i in range(0, min(len(full_content), 10000), chunk_size):
            chunk = full_content[i:i+chunk_size]
            if len(chunk) > 100:  # Only add substantial chunks
                chunks.append(chunk)
        
        # Store in FAISS
        vector_store.add_texts(chunks)
        
        return f"Successfully scraped and stored information from Wikipedia.\n\nTitle: {title}\n\nScraped {len(chunks)} chunks of content totaling {len(full_content)} characters.\n\nYou can now ask questions about this content using the query_knowledge tool."
        
    except httpx.HTTPError as e:
        error_msg = f"Error fetching Wikipedia page: {str(e)}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error scraping Wikipedia: {str(e)}"
        logger.error(error_msg)
        return error_msg
#endregion Wikipedia tool

def _build_messages(prompt: str):
    """Match the training/system format you used for the server."""
    global vc_selection
    return [
        {
            "role": "system",
            "content": "Convert the following natural language command to the correct voice control command format."
        },
        {
            "role": "user",
            "content": f"{prompt} | selection: {vc_selection}"
        },
    ]


@mcp.tool()
async def translate_to_vc(query: str) -> str:
    """
    Convert a natural-language voice command into the structured voice command.
    Speaks the translated command immediately on the server side.

    Args:
        query: The user's natural-language command.

    Returns:
        The translated voice command as a plain string.

    Raises:
        RuntimeError on empty model response.
    """
    global vc_selection
    messages = _build_messages(query)
    print(messages)
    model: str = "vc_finetuned"

    # Call the blocking Ollama client off the event loop
    try:
        resp = await asyncio.to_thread(
            _ollama_client.chat,
            model=model,
            messages=messages,
            options={
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
                "num_predict": 125,
            },
        )
    except Exception as e:
        raise RuntimeError(f"Ollama chat error: {e}. Ensure model '{model}' is loaded and {OLLAMA_HOST} is reachable.")

    content = (resp.get("message") or {}).get("content", "").strip()

    if 'SELECT' in content:
        vc_selection = content.split()[-1].strip()
    else:
        vc_selection = ""

    content_formatted = "##VC##" + content + "##VC##"
    print(">>>", content_formatted)

    if not content:
        raise RuntimeError(
            f"Empty response from model '{model}'. "
            f"Check that Ollama is running at {OLLAMA_HOST} and the model is available."
        )

    # Speak the translated command on server side
    speak_text(content)

    return content_formatted


@mcp.tool()
async def translate_to_cc(query: str) -> str:
    """
    Convert a natural-language voice command into the structured voice command when the command is provided with CC:
    Speaks the command immediately on the server side.

    Args:
        query: The user's natural-language command.

    Returns:
        The translated voice command as a plain string.

    Raises:
        RuntimeError on empty model response.
    """
    global vc_selection

    extracted_command = query.split(':')[-1].strip()
    content = "##VC##" + extracted_command + "##VC##"

    print("CC>>", content)

    # Speak the command on server side
    speak_text(extracted_command)

    return content


@mcp.tool()
async def speak_on_server(text: str) -> str:
    """
    Speak text on the server side using TTS.
    Useful for providing audio feedback without waiting for UI rendering.

    Args:
        text: Text to speak

    Returns:
        Confirmation message
    """
    try:
        speak_text(text)
        return f"Speaking: {text[:50]}..." if len(text) > 50 else f"Speaking: {text}"
    except Exception as e:
        error_msg = f"Error speaking text: {str(e)}"
        logger.error(error_msg)
        return error_msg

# @mcp.tool()
# async def translate_to_vc(query: str) -> str:
#     """
#     Convert a natural-language voice command into the structured voice command.
#
#     Args:
#         query: The user’s natural-language command.
#
#     Returns:
#         The translated voice command as a plain string.
#
#     Raises:
#         RuntimeError on empty model response.
#     """
#     global vc_selection
#     messages = _build_messages(query)
#     print(messages)
#     model: str = "vc_finetuned"
#
#     # Call the blocking Ollama client off the event loop
#     try:
#         resp = await asyncio.to_thread(
#             _ollama_client.chat,
#             model=model,
#             messages=messages,
#             options={
#                 "temperature": 1.0,
#                 "top_p": 0.95,
#                 "top_k": 64,
#                 "num_predict": 125,
#             },
#         )
#     except Exception as e:
#         # Keep it as a string return if you prefer not to raise
#         raise RuntimeError(f"Ollama chat error: {e}. Ensure model '{model}' is loaded and {OLLAMA_HOST} is reachable.")
#
#     content = (resp.get("message") or {}).get("content", "").strip()
#
#     if 'SELECT' in content:
#         vc_selection = content.split()[-1].strip()
#     else:
#         vc_selection = ""
#
#     content = "##VC##" + content + "##VC##"
#     print(">>>", content)
#     if not content:
#         raise RuntimeError(
#             f"Empty response from model '{model}'. "
#             f"Check that Ollama is running at {OLLAMA_HOST} and the model is available."
#         )
#
#     return content
#
#
# @mcp.tool()
# async def translate_to_cc(query: str) -> str:
#     """
#     Convert a natural-language voice command into the structured voice command. when the command is provided with CC:
#
#     Args:
#         query: The user’s natural-language command.
#
#     Returns:
#         The translated voice command as a plain string.
#
#     Raises:
#         RuntimeError on empty model response.
#     """
#     global vc_selection
#
#     content = "##VC##" + query.split(':')[-1] + "##VC##"
#
#     print("CC>>", content)
#
#     return content


# new tool
BASE_DIR = Path(__file__).resolve().parent   # folder where rag_server.py lives
TX_DIR = BASE_DIR / "my_files"   # <-- use subdirectory next to rag_server.py
TX_CHUNK = 1000

@mcp.tool()
async def ingest_tx() -> str:
    """
    Ingest local corpus into the vector DB.

    Reads all *.txt under ./my_files/, splits each file into ~TX_CHUNK-char
    pieces, prefixes chunks with their relative file path for provenance, and
    appends them to the global FAISS-backed `vector_store` so they are
    retrievable via `query_knowledge`.
    Returns a brief count summary.
    """
    files = sorted(TX_DIR.rglob("*.txt"))
    chunks: list[str] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        rel = f.relative_to(TX_DIR).as_posix()
        for i in range(0, len(text), TX_CHUNK):
            piece = text[i:i+TX_CHUNK].strip()
            if piece:
                chunks.append(f"[FILE] {rel}\n{piece}")
    if chunks:
        vector_store.add_texts(chunks)  # uses your lazy init + FAISS add
    return f"files={len(files)}, chunks={len(chunks)}, total_docs={len(vector_store.documents)}"

#region Knowledge query tools
@mcp.tool()
async def query_knowledge(question: str) -> str:
    """Query the knowledge base for information about previously scraped Wikipedia content.
    
    This tool searches the vector database for relevant content and returns an answer.
    You must scrape Wikipedia content first using the scrape_wikipedia tool.
    
    Args:
        question: Question to ask about the scraped content
    
    Returns:
        Answer based on stored Wikipedia content or error message
    """
    try:
        # Check if vector store has content
        if vector_store.index is None or len(vector_store.documents) == 0:
            return "Error: No information has been stored yet. Please scrape a Wikipedia page first using the scrape_wikipedia tool."
        
        logger.info(f"Querying knowledge base with question: {question}")
        
        # Retrieve relevant documents
        relevant_docs = vector_store.similarity_search(question, k=3)
        
        if not relevant_docs:
            return "I couldn't find relevant information to answer your question. The stored content may not contain information about this topic."
        
        # Combine relevant documents into context
        context = "\n\n---\n\n".join(relevant_docs)
        
        # Return the context - the LLM will use this to answer the question
        return f"Based on the stored Wikipedia content, here is the relevant information:\n\n{context}\n\n(This information can be used to answer the question: {question})"
        
    except Exception as e:
        error_msg = f"Error querying knowledge base: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
async def clear_knowledge_base() -> str:
    """Clear all stored Wikipedia content from the knowledge base.
    
    This removes all documents from the FAISS vector store, allowing you to start fresh
    with new Wikipedia pages.
    
    Returns:
        Confirmation message
    """
    try:
        vector_store.clear()
        return "Successfully cleared the knowledge base. You can now scrape new Wikipedia pages."
    except Exception as e:
        error_msg = f"Error clearing knowledge base: {str(e)}"
        logger.error(error_msg)
        return error_msg


# Server Status Tool
async def get_server_status() -> str:
    """Get the current status of the MCP server and vector database.

    Returns:
        Status information including number of stored documents
    """
    doc_count = len(vector_store.documents) if vector_store.documents else 0
    has_index = vector_store.index is not None
    is_initialized = vector_store.initialized

    status = f"""
        MCP Server Status:
        - Server: Running
        - Vector Store Initialized: {is_initialized}
        - FAISS Index Created: {has_index}
        - Stored Documents: {doc_count}
        - Available Tools: 
          * Arithmetic: add, subtract, multiply, divide
          * Voice Commands: translate_to_vc, translate_to_cc, execute_voice_command
          * Screen Analysis: parse_screen_with_ocr, find_text_number
          * Knowledge: scrape_wikipedia, query_knowledge, clear_knowledge_base, ingest_tx
          * Utilities: speak_on_server, get_server_status

        Ready to accept requests!
        """
    return status.strip()

#endregion knowledge query tools
import pyautogui
from PIL import Image, ImageGrab
from paddleocr import PaddleOCR
import pyttsx3
import threading

# Initialize OCR once
_ocr_instance = None
_tts_engine = None
_tts_lock = threading.Lock()


def get_ocr():
    """Lazy initialization of OCR model"""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _ocr_instance


_tts_lock = threading.Lock()


def speak_text(text_to_speak: str):
    """Speak text using TTS in a separate thread (server-side)"""

    def speak():
        try:
            # Acquire lock before initializing engine
            with _tts_lock:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.say(text_to_speak)
                engine.runAndWait()
                # Don't call engine.stop() - causes issues
                del engine  # Let garbage collector handle it

            logger.info(f"TTS spoke: {text_to_speak[:50]}...")
        except Exception as e:
            logger.error(f"TTS Error: {e}")

    tts_thread = threading.Thread(target=speak, daemon=True)
    tts_thread.start()


# @mcp.tool()
# async def take_screenshot() -> str:
#     """
#     Take a screenshot of the current screen and save it to a temporary file.
#
#     Returns:
#         Path to the saved screenshot file
#     """
#     try:
#         # Use a more reliable temporary file creation
#         temp_dir = tempfile.gettempdir()
#         screenshot_path = os.path.join(temp_dir, f"screenshot_{os.getpid()}_{int(asyncio.get_event_loop().time())}.png")
#
#         logger.info(f"Attempting to save screenshot to: {screenshot_path}")
#
#         # Use Pillow's ImageGrab which is more reliable
#         try:
#             screenshot = ImageGrab.grab()
#             screenshot.save(screenshot_path, 'PNG')
#             logger.info(f"Screenshot saved successfully to: {screenshot_path}")
#         except Exception as e:
#             logger.warning(f"ImageGrab failed: {e}, trying pyautogui...")
#             # Fallback to pyautogui
#             import pyautogui
#             screenshot = pyautogui.screenshot()
#             screenshot.save(screenshot_path, 'PNG')
#             logger.info(f"Screenshot saved via pyautogui to: {screenshot_path}")
#
#         # Verify the file actually exists
#         if not os.path.exists(screenshot_path):
#             raise FileNotFoundError(f"Screenshot file was not created at {screenshot_path}")
#
#         file_size = os.path.getsize(screenshot_path)
#         logger.info(f"Screenshot file size: {file_size} bytes")
#
#         return screenshot_path  # Return just the path, not a message
#
#     except Exception as e:
#         error_msg = f"Error taking screenshot: {str(e)}"
#         logger.error(error_msg)
#         return f"ERROR: {error_msg}"


def _take_screenshot_internal() -> str:
    """
    Internal function to take a screenshot and save it to a temporary file.
    Returns the path to the saved screenshot file.
    """
    try:
        # Create temporary file path
        temp_dir = '/Users/ibk5106/Desktop/IST_courses/TA/IST_597_003_AI_Private/rag_mcp/screenshots' #  tempfile.gettempdir()
        screenshot_path = os.path.join(
            temp_dir,
            f"screenshot_{os.getpid()}_{int(asyncio.get_event_loop().time())}.png"
        )

        logger.info(f"Taking screenshot, will save to: {screenshot_path}")

        # Use Pillow's ImageGrab which is more reliable
        try:
            screenshot = ImageGrab.grab()
            screenshot.save(screenshot_path, 'PNG')
            logger.info(f"Screenshot saved successfully to: {screenshot_path}")
        except Exception as e:
            logger.warning(f"ImageGrab failed: {e}, trying pyautogui...")
            # Fallback to pyautogui
            import pyautogui
            screenshot = pyautogui.screenshot()
            screenshot.save(screenshot_path, 'PNG')
            logger.info(f"Screenshot saved via pyautogui to: {screenshot_path}")

        # Verify the file actually exists
        if not os.path.exists(screenshot_path):
            raise FileNotFoundError(f"Screenshot file was not created at {screenshot_path}")

        file_size = os.path.getsize(screenshot_path)
        logger.info(f"Screenshot file size: {file_size} bytes")

        return screenshot_path

    except Exception as e:
        error_msg = f"Error taking screenshot: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


import re


@mcp.tool()
async def parse_screen_with_ocr() -> str:
    """
    Take a screenshot of the current screen and parse it using OCR to extract text and numbered labels.
    This function automatically handles screenshot capture and OCR parsing.

    Handles two number formats:
    1. Embedded numbers: "2chrome" -> extracts "2" and "chrome"
    2. Separate numbers: "2" near "Chrome" -> creates "2 Chrome"

    Returns:
        Markdown formatted string with detected numbers and their associated text
    """
    screenshot_path = None
    try:
        # Step 1: Take screenshot internally
        logger.info("Taking screenshot for OCR analysis...")
        screenshot_path = _take_screenshot_internal()

        # Step 2: Verify file exists
        if not os.path.exists(screenshot_path):
            error_msg = f"Screenshot file not found at: {screenshot_path}"
            logger.error(error_msg)
            return f"ERROR: {error_msg}"

        logger.info(f"Parsing screenshot at: {screenshot_path}")

        # Step 3: Run OCR
        ocr = get_ocr()
        result = ocr.predict(screenshot_path)

        # Step 4: Extract text and positions from PaddleOCR result
        all_detections = []

        # PaddleOCR returns a list with one dict containing all results
        if isinstance(result, list) and len(result) > 0:
            ocr_data = result[0]  # Get first (and only) result dict

            # Extract the arrays we need
            rec_texts = ocr_data.get('rec_texts', [])
            rec_scores = ocr_data.get('rec_scores', [])
            rec_polys = ocr_data.get('rec_polys', [])

            logger.info(f"Found {len(rec_texts)} text items in OCR result")

            # Combine text, scores, and bounding boxes
            for i, (text, score) in enumerate(zip(rec_texts, rec_scores)):
                # Get bounding box if available
                bbox = rec_polys[i].tolist() if i < len(rec_polys) else []

                # Calculate center position of bounding box
                if bbox and len(bbox) >= 4:
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    center_x = sum(x_coords) / len(x_coords)
                    center_y = sum(y_coords) / len(y_coords)
                else:
                    center_x, center_y = 0, 0

                # Only include items with reasonable confidence and non-empty text
                if text and text.strip() and score > 0.3:
                    all_detections.append({
                        'text': text.strip(),
                        'confidence': float(score),
                        'bbox': bbox,
                        'center_x': center_x,
                        'center_y': center_y
                    })

        # Step 5: Process detections to extract number-text mappings
        number_text_map = []
        processed_indices = set()

        # Method 1: Extract embedded numbers (e.g., "2chrome" -> "2" + "chrome")
        for i, det in enumerate(all_detections):
            text = det['text']
            # Check if text starts with digit(s) followed by non-digit characters
            match = re.match(r'^(\d+)([a-zA-Z].*)', text)
            if match:
                number = match.group(1)
                label = match.group(2)
                number_text_map.append({
                    'number': number,
                    'text': label,
                    'confidence': det['confidence'],
                    'source': 'embedded',
                    'position': (int(det['center_x']), int(det['center_y']))
                })
                processed_indices.add(i)
                logger.info(f"Found embedded number: {number}{label}")

        # Method 2: Find standalone numbers and match with nearby text
        standalone_numbers = []
        text_labels = []

        for i, det in enumerate(all_detections):
            if i in processed_indices:
                continue

            text = det['text']

            # Is it a pure number?
            if text.isdigit():
                standalone_numbers.append({
                    'index': i,
                    'number': text,
                    'center_x': det['center_x'],
                    'center_y': det['center_y'],
                    'confidence': det['confidence']
                })
            # Is it text (not starting with digit)?
            elif not text[0].isdigit():
                text_labels.append({
                    'index': i,
                    'text': text,
                    'center_x': det['center_x'],
                    'center_y': det['center_y'],
                    'confidence': det['confidence']
                })

        # Match standalone numbers with nearby text labels
        for num_det in standalone_numbers:
            num_value = num_det['number']
            num_x = num_det['center_x']
            num_y = num_det['center_y']

            # Find closest text (within 500 pixels)
            closest_text = None
            min_distance = float('inf')
            matched_index = None

            for text_det in text_labels:
                if text_det['index'] in processed_indices:
                    continue

                text_x = text_det['center_x']
                text_y = text_det['center_y']

                # Calculate distance (simple Euclidean)
                distance = ((num_x - text_x) ** 2 + (num_y - text_y) ** 2) ** 0.5

                # Numbers are usually close to their labels (within ~500px)
                if distance < 100 and distance < min_distance:
                    min_distance = distance
                    closest_text = text_det['text']
                    matched_index = text_det['index']

            if closest_text:
                number_text_map.append({
                    'number': num_value,
                    'text': closest_text,
                    'confidence': num_det['confidence'],
                    'source': 'distance',
                    'distance': int(min_distance),
                    'position': (int(num_x), int(num_y))
                })
                processed_indices.add(num_det['index'])
                if matched_index:
                    processed_indices.add(matched_index)
                logger.info(
                    f"Found standalone number {num_value} near '{closest_text}' (distance: {int(min_distance)}px)")

        # Sort by number value
        number_text_map.sort(key=lambda x: int(x['number']) if x['number'].isdigit() else 999)

        # Step 6: Save full data for debugging
        screenshot_dir = os.path.dirname(screenshot_path)
        screenshot_basename = os.path.splitext(os.path.basename(screenshot_path))[0]

        debug_data = {
            'screenshot_path': screenshot_path,
            'total_detections': len(all_detections),
            'number_text_mappings': number_text_map,
            'all_detections': all_detections,
            'processing_stats': {
                'embedded_numbers': len([m for m in number_text_map if m.get('source') == 'embedded']),
                'distance_matched': len([m for m in number_text_map if m.get('source') == 'distance']),
                'total_mapped': len(number_text_map)
            }
        }

        json_path = os.path.join(screenshot_dir, f"{screenshot_basename}_ocr.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved full OCR JSON to: {json_path}")

        # Step 7: Create concise markdown format for the model
        markdown_lines = [
            "# Screen Analysis - Numbered Elements\n",
            f"**Screenshot:** `{screenshot_path}`\n",
            f"**Total mappings:** {len(number_text_map)}\n",
            "\n## Number → Element Mapping:\n"
        ]

        if number_text_map:
            for item in number_text_map:
                # Format: **number** → "text" (source: embedded/distance)
                source_info = f"[{item['source']}]" if 'source' in item else ""
                markdown_lines.append(f"**{item['number']}** → \"{item['text']}\" {source_info}")
        else:
            markdown_lines.append("*No numbered elements found. Make sure 'show numbers' command was executed first.*")

        markdown_output = "\n".join(markdown_lines)

        # Step 8: Save markdown as text file for debugging
        txt_path = os.path.join(screenshot_dir, f"{screenshot_basename}_ocr.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(markdown_output)
        logger.info(f"Saved OCR text to: {txt_path}")

        logger.info(f"OCR complete: {len(number_text_map)} number-text mappings created")
        logger.info(f"Debug files: {screenshot_path}, {json_path}, {txt_path}")

        return markdown_output

    except Exception as e:
        error_msg = f"Error parsing screen with OCR: {str(e)}"
        logger.error(error_msg)
        logger.exception("Full traceback:")
        return f"ERROR: {error_msg}"


@mcp.tool()
async def find_text_number(ocr_result: str, target_text: str) -> str:
    """
    Find the number associated with a specific text in OCR results.
    Searches through the markdown formatted OCR results for the target text.

    Args:
        ocr_result: Markdown string from parse_screen_with_ocr
        target_text: The text to search for (e.g., "Home", "Settings")

    Returns:
        The number associated with the target text (e.g., "5")
    """
    try:
        # Parse the markdown to find number mappings
        # Look for lines like: **1** → "Home" [embedded]

        target_lower = target_text.lower()

        # Split by lines and look for mapping pattern
        lines = ocr_result.split('\n')

        for line in lines:
            # Check if line contains the mapping pattern: **N** → "text"
            if '→' in line and '**' in line:
                # Extract number and text
                parts = line.split('→')
                if len(parts) == 2:
                    # Extract number from **N**
                    number_part = parts[0].strip()
                    number = number_part.replace('*', '').strip()

                    # Extract text from "text" [source]
                    text_part = parts[1].strip()
                    # Remove source indicator like [embedded] or [distance]
                    text_part = re.sub(r'\[.*?\]', '', text_part).strip()
                    # Remove quotes
                    text_part = text_part.strip('"')

                    # Check if target text matches (case-insensitive, partial match)
                    if target_lower in text_part.lower():
                        logger.info(f"Found '{target_text}' associated with number: {number}")
                        return number  # Return just the number

        # If not found
        error_msg = f"Could not find '{target_text}' among numbered elements. Available elements: check OCR results."
        logger.warning(error_msg)
        return error_msg

    except Exception as e:
        error_msg = f"Error finding text number: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
async def execute_voice_command(command: str) -> str:
    """
    Execute a voice command by translating it and sending to the UI.
    This is a wrapper that can be called in multi-step workflows.
    Speaks the command immediately on the server side.

    Args:
        command: Natural language command or direct voice command

    Returns:
        Confirmation message with voice command marker
    """
    try:
        # If command already looks like a voice command, use it directly
        if any(keyword in command.upper() for keyword in ['CLICK', 'SELECT', 'SCROLL', 'SHOW']):
            formatted = f"##VC##${command}##VC##"
            logger.info(f"Executing direct voice command: {command}")
            print(f"Executing direct voice command: {command}")
            # Speak immediately on server side
            speak_text(command)

            return formatted
        else:
            # Otherwise translate it
            translated = await translate_to_cc(command)
            logger.info(f"Translated and executing: {translated}")
            print(f"Translated and executing: {translated}")

            # Extract the command between ##VC## markers and speak it
            if "##VC##" in translated:
                parts = translated.split("##VC##")
                if len(parts) >= 2:
                    voice_cmd = parts[1].strip()
                    speak_text(voice_cmd)

            return translated

    except Exception as e:
        error_msg = f"Error executing voice command: {str(e)}"
        logger.error(error_msg)
        return error_msg

# Main
if __name__ == "__main__":
    logger.info("Starting MCP Server with LLM Agent Tools...")
    logger.info("Available tools: arithmetic operations, Wikipedia scraping, knowledge querying")
    logger.info("Transport: streamable-http on port 3000")
    
    # Run the server with streamable-http transport
    mcp.run(transport="streamable-http")
