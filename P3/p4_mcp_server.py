"""
MCP Agent Server with Sequential Workflow Enforcement
Uses mcp-agent workflow decorators to guarantee execution order
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from mcp_agent.app import MCPApp
from mcp_agent.executor.workflow import Workflow, WorkflowResult
from datetime import timedelta
import torch
import tempfile
import json
import re
import logging
from PIL import ImageGrab
from paddleocr import PaddleOCR
import pyttsx3
import threading
from ollama import Client

# Setup
torch.set_default_device('cpu')
torch.set_num_threads(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP App
app = MCPApp(name="voice-command-agent")

# Ollama client
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
_ollama_client = Client(host=OLLAMA_HOST)

# TTS Engine
_tts_engine = None
_tts_lock = threading.Lock()

# Global vector store instanc

# region Arithmetic Tools for quick testing
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


# endregion arithmatic tools

# region Wikipedia Tool
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
            chunk = full_content[i:i + chunk_size]
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


# endregion Wikipedia tool

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


def _get_tts_engine():
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = pyttsx3.init()
        _tts_engine.setProperty('rate', 150)
    return _tts_engine


def speak_text(text: str):
    def speak():
        try:
            with _tts_lock:
                engine = _get_tts_engine()
                engine.say(text)
                engine.runAndWait()
            logger.info(f"TTS: {text[:50]}...")
        except Exception as e:
            logger.error(f"TTS Error: {e}")
            global _tts_engine
            _tts_engine = None

    threading.Thread(target=speak, daemon=True).start()


# OCR Instance
_ocr_instance = None


def get_ocr():
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


# Voice command selection
vc_selection = ""


# ============================================
# WORKFLOW DEFINITION (GUARANTEED ORDER)
# ============================================

@app.workflow
class VoiceCommandWorkflow(Workflow[dict]):
    """
    Sequential UI automation workflow.
    Steps execute in exact order: 1→2→3→4
    """

    @app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def step_1_show_numbers(self) -> dict:
        """Step 1: Show numbered labels on screen"""
        logger.info("Step 1: Showing numbered labels")

        command = "show numbers"
        result = f"##VC##${command}##VC##"

        speak_text(command)

        return {
            'step': 1,
            'command': result,
            'status': 'completed'
        }

    @app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def step_2_capture_and_analyze(self) -> dict:
        """Step 2: Take screenshot and analyze with OCR"""
        logger.info("Step 2: Capturing and analyzing screen")

        # Take screenshot
        temp_dir = "/Users/ibk5106/Desktop/IST_courses/TA/IST_597_003_AI_Private/P3/screenshots"  # tempfile.gettempdir()
        screenshot_path = os.path.join(
            temp_dir,
            f"screenshot_{os.getpid()}.png"
        )

        screenshot = ImageGrab.grab()
        screenshot.save(screenshot_path, 'PNG')
        logger.info(f"Screenshot: {screenshot_path}")

        # Run OCR
        ocr = get_ocr()
        result = ocr.predict(screenshot_path)

        # Parse detections
        all_detections = []
        if isinstance(result, list) and len(result) > 0:
            ocr_data = result[0]
            rec_texts = ocr_data.get('rec_texts', [])
            rec_scores = ocr_data.get('rec_scores', [])
            rec_polys = ocr_data.get('rec_polys', [])

            for i, (text, score) in enumerate(zip(rec_texts, rec_scores)):
                bbox = rec_polys[i].tolist() if i < len(rec_polys) else []

                if bbox and len(bbox) >= 4:
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    center_x = sum(x_coords) / len(x_coords)
                    center_y = sum(y_coords) / len(y_coords)
                else:
                    center_x, center_y = 0, 0

                if text and text.strip() and score > 0.3:
                    all_detections.append({
                        'text': text.strip(),
                        'center_x': center_x,
                        'center_y': center_y
                    })

        # Extract number-text mappings
        number_text_map = []
        processed = set()

        # Method 1: Embedded (e.g., "2chrome")
        for i, det in enumerate(all_detections):
            match = re.match(r'^(\d+)([a-zA-Z].*)', det['text'])
            if match:
                number_text_map.append({
                    'number': match.group(1),
                    'text': match.group(2)
                })
                processed.add(i)

        # Method 2: Distance matching
        numbers = [det for i, det in enumerate(all_detections) if i not in processed and det['text'].isdigit()]
        texts = [det for i, det in enumerate(all_detections) if i not in processed and not det['text'][0].isdigit()]

        for num_det in numbers:
            closest_text = None
            min_dist = float('inf')

            for text_det in texts:
                dist = ((num_det['center_x'] - text_det['center_x']) ** 2 +
                        (num_det['center_y'] - text_det['center_y']) ** 2) ** 0.5
                if dist < 500 and dist < min_dist:
                    min_dist = dist
                    closest_text = text_det['text']

            if closest_text:
                number_text_map.append({
                    'number': num_det['text'],
                    'text': closest_text
                })

        # Sort by number
        number_text_map.sort(key=lambda x: int(x['number']) if x['number'].isdigit() else 999)

        # Save debug files
        json_path = screenshot_path.replace('.png', '_ocr.json')
        with open(json_path, 'w') as f:
            json.dump({'mappings': number_text_map}, f, indent=2)

        # Create markdown
        markdown = "# Number → Element\n\n"
        for item in number_text_map:
            markdown += f"**{item['number']}** → \"{item['text']}\"\n"

        txt_path = screenshot_path.replace('.png', '_ocr.txt')
        with open(txt_path, 'w') as f:
            f.write(markdown)

        logger.info(f"Found {len(number_text_map)} mappings")

        return {
            'step': 2,
            'screenshot': screenshot_path,
            'mappings': number_text_map,
            'markdown': markdown,
            'status': 'completed'
        }

    @app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def step_3_find_element(self, ocr_data: dict, target: str) -> dict:
        """Step 3: Find target element number"""
        logger.info(f"Step 3: Finding '{target}'")

        target_lower = target.lower()
        mappings = ocr_data['mappings']

        for item in mappings:
            if target_lower in item['text'].lower():
                number = item['number']
                logger.info(f"Found '{target}' at number {number}")
                return {
                    'step': 3,
                    'target': target,
                    'number': number,
                    'status': 'completed'
                }

        raise ValueError(f"Could not find '{target}' in OCR results")

    @app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def step_4_click_element(self, find_data: dict) -> dict:
        """Step 4: Execute click command"""
        number = find_data['number']
        target = find_data['target']

        logger.info(f"Step 4: Clicking {target} at number {number}")

        command = f"click {number}"
        result = f"##VC##${command}##VC##"

        speak_text(command)

        return {
            'step': 4,
            'command': result,
            'target': target,
            'number': number,
            'status': 'completed'
        }

    @app.workflow_run
    async def run(self, task: str, target: str) -> WorkflowResult[dict]:
        """
        Main workflow execution - GUARANTEED sequential order.

        Args:
            task: Task description (e.g., "go to home")
            target: Target element to find (e.g., "home")
        """
        logger.info(f"Starting workflow: {task}")

        # Step 1: Show numbers (MUST complete first)
        step1_result = await self.step_1_show_numbers()
        logger.info(f"Step 1 done: {step1_result}")

        # Step 2: Analyze screen (ONLY runs after step 1)
        step2_result = await self.step_2_capture_and_analyze()
        logger.info(f"Step 2 done: Found {len(step2_result['mappings'])} elements")

        # Step 3: Find target (ONLY runs after step 2)
        step3_result = await self.step_3_find_element(step2_result, target)
        logger.info(f"Step 3 done: Found at {step3_result['number']}")

        # Step 4: Click (ONLY runs after step 3)
        step4_result = await self.step_4_click_element(step3_result)
        logger.info(f"Step 4 done: Clicked {step4_result['target']}")

        return WorkflowResult(value={
            'task': task,
            'target': target,
            'steps': [step1_result, step2_result, step3_result, step4_result],
            'status': 'workflow_completed',
            'message': f"Successfully navigated to {target}!"
        })


# ============================================
# SIMPLE TOOLS (for /VC and /CC)
# ============================================

@app.tool()
def translate_vc(command: str) -> str:
    """Translate natural language voice command"""
    # Use your vc_finetuned model here
    result = f"##VC##${command}##VC##"
    speak_text(command)
    return result


@app.tool()
def translate_cc(command: str) -> str:
    """Process direct voice command"""
    result = f"##VC##${command}##VC##"
    speak_text(command)
    return result


# ============================================
# MAIN EXECUTION TOOL
# ============================================

@app.tool()
async def execute_ui_task(task_description: str, target_element: str) -> str:
    """
    Execute a multi-step UI navigation task.
    This automatically runs all 4 steps in order.

    Args:
        task_description: What to do (e.g., "go to home page")
        target_element: What to click (e.g., "home")

    Returns:
        JSON with workflow results
    """
    # Run the workflow - execution order is GUARANTEED by mcp-agent
    workflow = VoiceCommandWorkflow()
    result = await workflow.run(task_description, target_element)

    return json.dumps(result.value, indent=2)


if __name__ == "__main__":
    # Run the MCP server
    app.run()