"""
Hybrid Architecture:
- mcp-agent: Workflow orchestration with guaranteed sequential execution
- FastMCP: Streamable HTTP transport on port 3000
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_default_device('cpu')
torch.set_num_threads(1)

from mcp.server.fastmcp import FastMCP
from mcp_agent.app import MCPApp
from mcp_agent.executor.workflow import Workflow, WorkflowResult
from datetime import timedelta

import tempfile
import json
import re
import logging
from PIL import ImageGrab
from paddleocr import PaddleOCR
from gtts import gTTS
from playsound import playsound
import os
import tempfile
import threading
from typing import List
from gtts import gTTS
from playsound import playsound
import os
import tempfile
import pyautogui

screen_width, screen_height = pyautogui.size()
print(f"Screen size (pyautogui): {screen_width}x{screen_height}")

# Screenshot size
screenshot = ImageGrab.grab()
img_width, img_height = screenshot.size
print(f"Screenshot size: {img_width}x{img_height}")

# Calculate scaling factor
scale_x = screen_width / img_width
scale_y = screen_height / img_height
print(f"Scaling factors: x={scale_x}, y={scale_y}")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# INITIALIZE BOTH: FastMCP + mcp-agent
# ============================================

# FastMCP for streamable HTTP transport
mcp_server = FastMCP("voice-command-workflows", port=3000)

# mcp-agent for workflow orchestration
mcp_agent_app = MCPApp(name="workflow-engine")

def speak_text(text: str):
    """Speak text using Google TTS (cross-platform, reliable)"""

    def speak():
        try:
            print(f">>> TTS: Speaking: {text[:50]}...")

            # Generate audio file
            temp_dir = tempfile.gettempdir()
            audio_file = os.path.join(temp_dir, f"tts_{hash(text)}.mp3")

            tts = gTTS(text=text, lang='en')
            tts.save(audio_file)

            # Play audio
            playsound(audio_file)

            # Clean up
            try:
                os.remove(audio_file)
            except:
                pass

            print(f">>> TTS: Done speaking")

        except Exception as e:
            print(f">>> TTS Error: {e}")

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


# ============================================
# MCP-AGENT WORKFLOWS (Guaranteed Sequential)
# ============================================

@mcp_agent_app.workflow
class SimpleNavigationWorkflow(Workflow[dict]):
    """Workflow: show numbers → click"""

    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def show_numbers(self) -> dict:
        """Step: Show numbered labels"""
        logger.info("Workflow Step: Show numbers")
        command = "show numbers"
        result = f"##VC##${command}##VC##"
        speak_text(command)
        return {'command': result, 'status': 'completed'}

    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def click_element(self, number: str) -> dict:
        """Step: Click element"""
        logger.info(f"Workflow Step: Click {number}")
        command = f"click {number}"
        result = f"##VC##${command}##VC##"
        speak_text(command)
        return {'command': result, 'number': number, 'status': 'completed'}

    @mcp_agent_app.workflow_run
    async def run(self, number: str) -> WorkflowResult[dict]:
        """Execute: show numbers → click"""
        logger.info(f"Simple Navigation: Click {number}")

        step1 = await self.show_numbers()
        step2 = await self.click_element(number)

        return WorkflowResult(value={
            'workflow': 'simple_navigation',
            'steps': [step1, step2],
            'status': 'completed',
            'message': f'Clicked number {number}'
        })


@mcp_agent_app.workflow
class SearchAndClickWorkflow(Workflow[dict]):
    """Workflow: show numbers → OCR → find → click (using mouse)"""

    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def show_numbers(self) -> dict:
        """Step: Show numbered labels"""
        logger.info("Workflow Step: Show numbers")
        command = "show numbers"
        result = f"##VC##${command}##VC##"
        speak_text(command)
        return {'command': result, 'status': 'completed'}

    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def analyze_screen(self) -> dict:
        """Step: Capture screenshot and run OCR"""
        logger.info("Workflow Step: Analyze screen")

        temp_dir = "/Users/ibk5106/Desktop/IST_courses/TA/IST_597_003_AI_Private/rag_mcp/screenshots"
        screenshot_path = os.path.join(temp_dir, f"screenshot_{id(self)}.png")
        screenshot = ImageGrab.grab()
        screenshot.save(screenshot_path, 'PNG')

        ocr = get_ocr()
        result = ocr.predict(screenshot_path)

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

        number_text_map = []
        processed = set()

        # Embedded numbers
        for i, det in enumerate(all_detections):
            match = re.match(r'^(\d+)([a-zA-Z].*)', det['text'])
            if match:
                number_text_map.append({
                    'number': match.group(1),
                    'text': match.group(2),
                    'center_x': det['center_x'],
                    'center_y': det['center_y']
                })
                processed.add(i)

        # Distance matching
        numbers = [det for i, det in enumerate(all_detections)
                   if i not in processed and det['text'].isdigit()]
        texts = [det for i, det in enumerate(all_detections)
                 if i not in processed and not det['text'][0].isdigit()]

        for num_det in numbers:
            closest_text = None
            closest_text_det = None
            min_dist = 99999

            for text_det in texts:
                dist = ((num_det['center_x'] - text_det['center_x']) ** 2 +
                        (num_det['center_y'] - text_det['center_y']) ** 2) ** 0.5
                if dist < 500 and dist < min_dist:
                    min_dist = dist
                    closest_text = text_det['text']
                    closest_text_det = text_det

            if closest_text and closest_text_det:
                number_text_map.append({
                    'number': num_det['text'],
                    'text': closest_text,
                    'center_x': closest_text_det['center_x'],  # Use text position
                    'center_y': closest_text_det['center_y']
                })

        number_text_map.sort(key=lambda x: int(x['number']) if x['number'].isdigit() else 999)

        json_path = screenshot_path.replace('.png', '_ocr.json')
        with open(json_path, 'w') as f:
            json.dump({'mappings': number_text_map}, f, indent=2)

        print(f"Found {len(number_text_map)} mappings")
        logger.info(f"Found {len(number_text_map)} mappings")

        return {
            'mappings': number_text_map,
            'screenshot': screenshot_path,
            'status': 'completed'
        }

    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def find_target(self, ocr_data: dict, target: str) -> dict:
        """Step: Find target element with coordinates"""
        logger.info(f"Workflow Step: Find '{target}'")

        target_lower = target.lower()
        for item in ocr_data['mappings']:
            if target_lower in item['text'].lower():
                number = item['number']
                center_x = item.get('center_x', 0)
                center_y = item.get('center_y', 0)

                logger.info(f"Found '{target}' at number {number}, position ({center_x}, {center_y})")
                print(f"Found '{target}' at number {number}, position ({center_x}, {center_y})")

                return {
                    'target': target,
                    'number': number,
                    'x': int(center_x),
                    'y': int(center_y),
                    'status': 'completed'
                }

        raise ValueError(f"Could not find '{target}'")

    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def click_element(self, find_data: dict) -> dict:
        """Step: Click element using mouse automation with coordinate scaling"""
        number = find_data['number']
        target = find_data['target']
        ocr_x = find_data['x']
        ocr_y = find_data['y']

        logger.info(f"Workflow Step: Click {target} at number {number}, OCR coordinates ({ocr_x}, {ocr_y})")
        print(f"Workflow Step: Click {target} at number {number}, OCR coordinates ({ocr_x}, {ocr_y})")

        try:
            # Get screen and screenshot dimensions
            screen_width, screen_height = pyautogui.size()
            screenshot = ImageGrab.grab()
            img_width, img_height = screenshot.size

            # Calculate scaling factors
            scale_x = screen_width / img_width
            scale_y = screen_height / img_height

            # Convert OCR coordinates to actual screen coordinates
            actual_x = int(ocr_x * scale_x)
            actual_y = int(ocr_y * scale_y)

            logger.info(f"Screen size: {screen_width}x{screen_height}")
            logger.info(f"Screenshot size: {img_width}x{img_height}")
            logger.info(f"Scaling factors: x={scale_x:.2f}, y={scale_y:.2f}")
            logger.info(f"Converted coordinates: ({actual_x}, {actual_y})")

            print(f"Screen size: {screen_width}x{screen_height}")
            print(f"Screenshot size: {img_width}x{img_height}")
            print(f"Scaling factors: x={scale_x:.2f}, y={scale_y:.2f}")
            print(f"Converted coordinates: ({actual_x}, {actual_y})")

            # Move mouse to position (visual feedback)
            pyautogui.moveTo(actual_x, actual_y, duration=0.3)

            # Small delay to see where mouse landed
            import time
            time.sleep(0.2)

            # Click
            pyautogui.click(actual_x, actual_y)

            speak_text(f"clicked {target}")

            logger.info(f"Successfully clicked at screen position ({actual_x}, {actual_y})")
            print(f"Successfully clicked at screen position ({actual_x}, {actual_y})")

            return {
                'action': 'mouse_click',
                'number': number,
                'target': target,
                'ocr_x': ocr_x,
                'ocr_y': ocr_y,
                'screen_x': actual_x,
                'screen_y': actual_y,
                'scale_x': scale_x,
                'scale_y': scale_y,
                'status': 'completed'
            }

        except Exception as e:
            logger.error(f"Failed to click: {e}")
            print(f"Failed to click: {e}")
            raise

    @mcp_agent_app.workflow_run
    async def run(self, target: str) -> WorkflowResult[dict]:
        """Execute: show numbers → OCR → find → click (with mouse)"""
        logger.info(f"Search and Click: {target}")

        step1 = await self.show_numbers()
        step2 = await self.analyze_screen()
        step3 = await self.find_target(step2, target)
        step4 = await self.click_element(step3)

        return WorkflowResult(value={
            'workflow': 'search_and_click',
            'target': target,
            'steps': [step1, step2, step3, step4],
            'status': 'completed',
            'message': f"Successfully clicked on {target} using mouse automation!"
        })

# ============================================
# FASTMCP TOOLS (Expose workflows via HTTP)
# ============================================

@mcp_server.tool()
async def click_number(number: str) -> str:
    """
    Simple workflow: Click a specific number.
    Uses mcp-agent workflow for guaranteed sequential execution.

    Args:
        number: Number to click (e.g., "5")
    """
    try:
        workflow = SimpleNavigationWorkflow()
        result = await workflow.run(number)
        return json.dumps(result.value, indent=2)
    except Exception as e:
        logger.error(f"Error: {e}")
        return json.dumps({
            'status': 'error',
            'error': str(e)
        })


@mcp_server.tool()
async def navigate_to(target: str) -> str:
    """
    Standard workflow: Navigate to target element by text.
    Uses mcp-agent workflow for guaranteed sequential execution.

    Args:
        target: Element text to find (e.g., "home", "settings")
    """
    try:
        workflow = SearchAndClickWorkflow()
        result = await workflow.run(target)
        return json.dumps(result.value, indent=2)
    except Exception as e:
        logger.error(f"Error: {e}")
        return json.dumps({
            'status': 'error',
            'error': str(e)
        })

@mcp_server.tool()
def translate_vc(command: str) -> str:
    """Single voice command translation"""
    result = f"##VC##${command}##VC##"
    speak_text(command)
    return result


@mcp_server.tool()
def translate_cc(command: str) -> str:
    """Direct voice command"""
    result = f"##VC##${command}##VC##"
    speak_text(command)
    return result


@mcp_server.tool()
def get_server_status() -> str:
    """Server status"""
    return """
Voice Command Workflows Server

Architecture:
- Transport: FastMCP streamable-http (port 3000)
- Workflow Engine: mcp-agent (guaranteed sequential execution)
- URL: http://127.0.0.1:3000/mcp

Available Workflows:
1. click_number(number) - Simple workflow
2. navigate_to(target) - Standard workflow  
3. navigate_multi_step(targets) - Complex workflow

Examples:
- click_number("5")
- navigate_to("home")
- navigate_multi_step(["menu", "settings"])

All workflows use mcp-agent for guaranteed execution order.
"""


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Hybrid Server: mcp-agent + FastMCP Streamable HTTP")
    logger.info("=" * 60)
    logger.info("Workflow Engine: mcp-agent (guaranteed sequential execution)")
    logger.info("Transport: FastMCP streamable-http")
    logger.info("Port: 3000")
    logger.info("URL: http://127.0.0.1:3000/mcp")
    logger.info("=" * 60)

    # Run FastMCP with streamable-http transport
    mcp_server.run(transport="streamable-http")