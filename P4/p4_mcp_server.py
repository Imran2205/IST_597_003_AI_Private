"""
Hybrid MCP Architecture for Voice-Controlled UI Automation

This module implements a hybrid architecture combining:
- mcp-agent: Provides workflow orchestration with guaranteed sequential execution
- FastMCP: Provides streamable HTTP transport on port 3000

The system enables voice-controlled screen interaction by:
1. Capturing screenshots of the current screen
2. Running OCR to detect UI elements and their positions
3. Mapping numbered labels to UI elements
4. Executing mouse clicks on target elements based on voice commands

Dependencies:
    - mcp.server.fastmcp: FastMCP server for HTTP transport
    - mcp_agent: Workflow orchestration framework
    - paddleocr: OCR engine for text detection
    - pyautogui: Mouse automation
    - PIL.ImageGrab: Screenshot capture
    - gtts/playsound: Text-to-speech feedback
    - pyttsx3: Alternative TTS engine

"""

import os

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================
# Set threading limits before importing numeric libraries to prevent
# thread contention issues with OCR and other compute-intensive operations
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import torch

# Configure PyTorch for single-threaded CPU execution
# This prevents resource contention with other libraries
torch.set_default_device('cpu')
torch.set_num_threads(1)

# =============================================================================
# IMPORTS
# =============================================================================
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
import threading
from typing import List
from pathlib import Path
import pyautogui
import pyttsx3


# =============================================================================
# TEXT-TO-SPEECH UTILITIES
# =============================================================================

def speak_text(text: str) -> None:
    """
    Speak text aloud using Google Text-to-Speech (gTTS).
    
    Runs asynchronously in a daemon thread to avoid blocking the main workflow.
    Audio is saved to a temporary file, played, then cleaned up.
    
    Args:
        text: The text string to be spoken aloud.
        
    Note:
        - Uses Google TTS which requires internet connectivity
        - Audio files are saved to system temp directory
        - Thread is daemonized so it won't prevent program exit
    """
    def speak():
        try:
            print(f">>> TTS: Speaking: {text[:50]}...")

            # Generate audio file in temp directory
            temp_dir = tempfile.gettempdir()
            audio_file = os.path.join(temp_dir, f"tts_{hash(text)}.mp3")

            # Convert text to speech and save
            tts = gTTS(text=text, lang='en')
            tts.save(audio_file)

            # Play the generated audio
            playsound(audio_file)

            # Clean up temporary audio file
            try:
                os.remove(audio_file)
            except:
                pass

        except Exception as e:
            print(f">>> TTS Error: {e}")

    # Run TTS in background thread to avoid blocking
    threading.Thread(target=speak, daemon=True).start()


# =============================================================================
# SCREEN DIMENSION DETECTION
# =============================================================================
# Detect screen dimensions and calculate scaling factors between
# actual screen resolution and screenshot resolution.
# This is crucial for accurate mouse click positioning.

screen_width, screen_height = pyautogui.size()
print(f"Screen size (pyautogui): {screen_width}x{screen_height}")

# Capture a screenshot to determine its resolution
screenshot = ImageGrab.grab()
img_width, img_height = screenshot.size
print(f"Screenshot size: {img_width}x{img_height}")

# Calculate scaling factors for coordinate conversion
# These are used to convert OCR coordinates to actual screen coordinates
scale_x = screen_width / img_width
scale_y = screen_height / img_height
print(f"Scaling factors: x={scale_x}, y={scale_y}")


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# MCP SERVER AND AGENT INITIALIZATION
# =============================================================================

# FastMCP server for streamable HTTP transport
# Provides the communication layer for MCP protocol over HTTP
mcp_server = FastMCP("P4", port=3000)

# mcp-agent application for workflow orchestration
# Ensures guaranteed sequential execution of workflow steps
mcp_agent_app = MCPApp(name="workflow-engine")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_env_file() -> dict:
    """
    Load environment variables from a .env file in the current directory.
    
    Parses a standard .env file format where each line contains KEY=VALUE pairs.
    Handles comments (lines starting with #) and quoted values.
    
    Returns:
        dict: A dictionary mapping environment variable names to their values.
              Returns empty dict if .env file doesn't exist.
              
    Example .env format:
        # This is a comment
        PATH_TO_SCREENSHOT="/path/to/screenshots"
        API_KEY='secret_key'
    """
    env_path = Path(".env")
    env_vars = {}

    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove surrounding quotes from values
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")

    return env_vars


# =============================================================================
# OCR SINGLETON
# =============================================================================

# Global OCR instance for lazy initialization
_ocr_instance = None


def get_ocr() -> PaddleOCR:
    """
    Get or create a singleton PaddleOCR instance.
    
    Uses lazy initialization to avoid loading OCR models until needed.
    Configures OCR for mobile-optimized models with minimal preprocessing
    for faster performance.
    
    Returns:
        PaddleOCR: Configured OCR instance ready for text detection/recognition.
        
    Note:
        - Uses PP-OCRv5 mobile models for balance of speed and accuracy
        - Disables document orientation and unwarping for UI screenshots
        - Thread-safe due to Python's GIL, but instance is reused
    """
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


def extract_ascii_digits(text: str) -> str:
    """
    Extract only ASCII digits (0-9) from a text string.
    
    Filters out non-ASCII numeric characters that might appear in OCR results,
    ensuring only standard digits are used for number matching.
    
    Args:
        text: Input string potentially containing mixed characters.
        
    Returns:
        str: String containing only ASCII digits from the input.
        
    Example:
        >>> extract_ascii_digits("123abc456")
        '123456'
        >>> extract_ascii_digits("①②③")
        ''
    """
    return ''.join(c for c in text if c.isascii() and c.isdigit())


# =============================================================================
# WORKFLOW: VanillaWorkflow
# =============================================================================

@mcp_agent_app.workflow
class VanillaWorkflow(Workflow[dict]):
    """
    Basic workflow for screen capture and status announcement.
    
    This workflow performs a simple three-step process:
    1. Load environment configuration (screenshot path)
    2. Capture a screenshot and save it
    3. Announce completion status via TTS
    
    Attributes:
        path_to_screenshot (str): Directory path where screenshots are saved.
    """
    
    path_to_screenshot = ''    

    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=1))
    async def read_path_to_screenshot(self) -> dict:
        """
        Load screenshot path from environment configuration.
        
        Reads the PATH_TO_SCREENSHOT variable from the .env file and stores
        it in the workflow instance for use by subsequent steps.
        
        Returns:
            dict: All environment variables loaded from .env file.
        """
        env_vars = load_env_file()
        self.path_to_screenshot = env_vars.get("PATH_TO_SCREENSHOT")        
        logger.info(f"screenshot path: {self.path_to_screenshot}")
        return env_vars

    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def take_screenshot(self) -> str:
        """
        Capture and save a screenshot of the current screen.
        
        Uses PIL's ImageGrab to capture the entire screen and saves it as
        a PNG file in the configured screenshot directory.
        
        Returns:
            str: Full path to the saved screenshot file, or empty string on failure.
        """
        current_screenshot_path = ''
        if self.path_to_screenshot:
            # Generate unique filename using object id
            current_screenshot_path = os.path.join(
                self.path_to_screenshot, 
                f"screenshot_{id(self)}.png"
            )
            screenshot = ImageGrab.grab()
            screenshot.save(current_screenshot_path, 'PNG')
            logger.info(f"A new screenshot is captured at {current_screenshot_path}")                        
        else:
            logger.error("The .env is not loaded. Unable to save the screenshot")

        return current_screenshot_path

    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def read_a_command_loadly(self, command: str) -> dict:
        """
        Announce a command or status message via text-to-speech.
        
        Args:
            command: The text to be spoken aloud.
            
        Returns:
            dict: Status dictionary with the command and completion status.
        """
        speak_text(command)
        return {'command': command, 'status': 'completed'}
        
        
    @mcp_agent_app.workflow_run
    async def run(self, target: str) -> WorkflowResult[dict]:
        """
        Execute the complete vanilla workflow.
        
        Orchestrates the sequential execution of all workflow steps:
        environment loading, screenshot capture, and status announcement.
        
        Args:
            target: Unused parameter (maintained for interface consistency).
            
        Returns:
            WorkflowResult[dict]: Result containing workflow metadata and status.
        """
        step1 = await self.read_path_to_screenshot()
        step2 = await self.take_screenshot()

        # wait for 0.2 second for safety
        time.sleep(0.2)

        status = 'completed' if step2 else 'failure'
        step3 = await self.read_a_command_loadly(status)
        
        return WorkflowResult(value={
            'workflow': 'VanillaWorkflow',
            'target': 'None',
            'steps': [step1, step2, step3],
            'status': status,
            'message': f"The workflow is completed {'successfully' if step2 else 'unsuccessfully'}"
        })


@mcp_server.tool()
async def vanilla_workflow_tool(target=None) -> str:
    """
    MCP tool endpoint for executing the VanillaWorkflow.
    
    Exposes the VanillaWorkflow as an MCP tool that can be invoked
    via the FastMCP HTTP interface.
    
    Args:
        target: Optional target parameter (unused, for interface consistency).
        
    Returns:
        str: JSON-formatted result of the workflow execution.
    """
    try:
        workflow = VanillaWorkflow()
        result = await workflow.run(target)
        return json.dumps(result.value, indent=2)
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return json.dumps({
            'status': 'error',
            'error': str(e)
        })


# =============================================================================
# WORKFLOW: ClickWorkFlow
# =============================================================================

@mcp_agent_app.workflow
class ClickWorkFlow(Workflow[dict]):
    """
    Workflow for clicking on a specific UI element by name.
    
    This workflow:
    1. Reads OCR metadata from a previously generated JSON file
    2. Searches for the target element by text matching
    3. Clicks on the element using scaled screen coordinates
    4. Announces the action via TTS
    """
    
    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=1))
    async def read_metadata(self, path_to_json_file: str) -> dict:
        """
        Load OCR metadata from a JSON file.
        
        Args:
            path_to_json_file: Path to the JSON file containing OCR mappings.
            
        Returns:
            dict: Parsed OCR data with element mappings and coordinates.
        """
        with open(path_to_json_file, 'r') as f:
            ocr_data = json.load(f)
        return ocr_data

    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def find_target(self, ocr_data: dict, target: str) -> dict:
        """
        Find a target UI element in the OCR data by text matching.
        
        Performs case-insensitive substring matching to locate the target
        element and retrieves its associated number label and coordinates.
        
        Args:
            ocr_data: Dictionary containing OCR mappings with 'mappings' key.
            target: Text string to search for in detected UI elements.
            
        Returns:
            dict: Target information including number, coordinates, and status.
            
        Raises:
            ValueError: If the target element cannot be found in the OCR data.
        """
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
        """
        Click on a UI element using mouse automation with coordinate scaling.
        
        Converts OCR coordinates (from screenshot resolution) to actual screen
        coordinates, moves the mouse to the target position, and performs a click.
        
        Args:
            find_data: Dictionary containing target info with 'number', 'target',
                      'x', and 'y' keys.
                      
        Returns:
            dict: Detailed click result including original and scaled coordinates,
                  scaling factors, and completion status.
                  
        Raises:
            Exception: If the click operation fails.
        """
        number = find_data['number']
        target = find_data['target']
        ocr_x = find_data['x']
        ocr_y = find_data['y']

        logger.info(f"Workflow Step: Click {target} at number {number}, OCR coordinates ({ocr_x}, {ocr_y})")
        print(f"Workflow Step: Click {target} at number {number}, OCR coordinates ({ocr_x}, {ocr_y})")

        try:
            # Get current screen and screenshot dimensions for scaling
            screen_width, screen_height = pyautogui.size()
            screenshot = ImageGrab.grab()
            img_width, img_height = screenshot.size

            # Calculate scaling factors to convert OCR coords to screen coords
            scale_x = screen_width / img_width
            scale_y = screen_height / img_height

            # Apply scaling to get actual screen coordinates
            actual_x = int(ocr_x * scale_x)
            actual_y = int(ocr_y * scale_y)

            # Log coordinate transformation details
            logger.info(f"Screen size: {screen_width}x{screen_height}")
            logger.info(f"Screenshot size: {img_width}x{img_height}")
            logger.info(f"Scaling factors: x={scale_x:.2f}, y={scale_y:.2f}")
            logger.info(f"Converted coordinates: ({actual_x}, {actual_y})")

            print(f"Screen size: {screen_width}x{screen_height}")
            print(f"Screenshot size: {img_width}x{img_height}")
            print(f"Scaling factors: x={scale_x:.2f}, y={scale_y:.2f}")
            print(f"Converted coordinates: ({actual_x}, {actual_y})")

            # Move mouse with animation for visual feedback
            pyautogui.moveTo(actual_x, actual_y, duration=0.3)

            # Brief pause to observe mouse position
            time.sleep(0.2)

            # Perform the click
            pyautogui.click(actual_x, actual_y)
            
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
        
    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def read_a_command_loadly(self, command: str) -> dict:
        """
        Announce a command or status via text-to-speech.
        
        Args:
            command: The text to be spoken aloud.
            
        Returns:
            dict: Status dictionary with the command and completion status.
        """
        speak_text(command)
        return {'command': command, 'status': 'completed'}
        
        
    @mcp_agent_app.workflow_run
    async def run(self, target: str, path_to_json: str) -> WorkflowResult[dict]:
        """
        Execute the complete click workflow.
        
        Orchestrates: metadata loading -> target finding -> clicking -> announcement.
        
        Args:
            target: Name of the UI element to click.
            path_to_json: Path to the JSON file with OCR mappings.
            
        Returns:
            WorkflowResult[dict]: Result containing workflow execution details.
        """
        step1 = await self.read_metadata(path_to_json)
        step2 = await self.find_target(step1, target)        
        step3 = await self.click_element(step2)
        step4 = await self.read_a_command_loadly(f'clicked on {target}')

        status = 'completed'

        return WorkflowResult(value={
            'workflow': 'VanillaWorkflow',
            'target': 'None',
            'steps': [step1, step2, step3],
            'status': status,
            'message': f"The workflow is completed"
        })


@mcp_server.tool()
async def click_workflow_tool(target: str) -> str:
    """
    MCP tool endpoint for clicking on a UI element by name.
    
    Reads environment configuration to locate the screenshot directory,
    loads the metadata JSON file, and executes the ClickWorkFlow to
    find and click the target element.
    
    Args:
        target: Name of the UI element to click (matched against OCR text).
        
    Returns:
        str: JSON-formatted result of the workflow execution.
    """
    try:
        # Load environment to get screenshot directory
        env_vars = load_env_file()
        path_to_screenshot = env_vars.get("PATH_TO_SCREENSHOT")   
        meta_json_path = os.path.join(path_to_screenshot, f"meta.json")
        
        # Read the path to the latest OCR JSON from metadata
        with open(meta_json_path, 'r') as f:
            json_path = dict(json.load(f)).get('json_path')
            print(f"json path {json_path}")

        # Execute the click workflow
        workflow2 = ClickWorkFlow()
        result = await workflow2.run(target=target, path_to_json=json_path)
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return json.dumps({
            'status': 'error',
            'error': str(e)
        })


# =============================================================================
# WORKFLOW: CaptureScreenWithNumbers
# =============================================================================

@mcp_agent_app.workflow
class CaptureScreenWithNumbers(Workflow[dict]):
    """
    Comprehensive screen capture workflow with OCR and number mapping.
    
    This workflow:
    1. Prepares the screen (announces listening start)
    2. Triggers numbered label display
    3. Captures screenshot and runs OCR
    4. Maps detected numbers to UI element text
    5. Saves mappings to JSON for later use by click workflows
    
    The OCR mapping handles two patterns:
    - Embedded numbers: "1Settings" -> number=1, text="Settings"
    - Adjacent numbers: Number "1" near text "Settings" matched by distance
    """

    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def prepare_screen(self, command: str) -> dict:
        """
        Announce a preparation command via TTS.
        
        Args:
            command: The command to announce (e.g., "start listening").
            
        Returns:
            dict: Command result with special voice command format.
        """
        result = f"##VC##${command}##VC##"
        speak_text(command)
        return {'command': result, 'status': 'completed'}

    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def analyze_screen(self) -> dict:
        """
        Capture screenshot and perform OCR with number-to-text mapping.
        
        This is the core analysis step that:
        1. Captures a screenshot of the current screen
        2. Runs PaddleOCR to detect all text elements
        3. Creates mappings between numbered labels and UI text
        4. Handles both embedded numbers (e.g., "1Settings") and 
           spatially adjacent number-text pairs
        5. Saves mappings to JSON for subsequent click operations
        
        Returns:
            dict: Analysis results including:
                - mappings: List of {number, text, center_x, center_y} dicts
                - screenshot_path: Path to saved screenshot
                - json_path: Path to saved OCR mappings JSON
                - status: Completion status
        """
        logger.info("Workflow Step: Analyze screen")

        # Load screenshot path from environment
        env_vars = load_env_file()
        temp_dir = env_vars.get("PATH_TO_SCREENSHOT")        
        logger.info(f"screenshot path: {temp_dir}")        
        
        # Define output file paths
        screenshot_path = os.path.join(temp_dir, f"screenshot_{id(self)}.png")
        meta_json_path = os.path.join(temp_dir, f"meta.json")

        # Capture and save screenshot
        screenshot = ImageGrab.grab()
        screenshot.save(screenshot_path, 'PNG')

        # Run OCR on the screenshot
        ocr = get_ocr()
        result = ocr.predict(screenshot_path)

        # Parse OCR results into detection list
        all_detections = []
        if isinstance(result, list) and len(result) > 0:
            ocr_data = result[0]
            rec_texts = ocr_data.get('rec_texts', [])
            rec_scores = ocr_data.get('rec_scores', [])
            rec_polys = ocr_data.get('rec_polys', [])

            for i, (text, score) in enumerate(zip(rec_texts, rec_scores)):
                bbox = rec_polys[i].tolist() if i < len(rec_polys) else []

                # Calculate center point of bounding box
                if bbox and len(bbox) >= 4:
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    center_x = sum(x_coords) / len(x_coords)
                    center_y = sum(y_coords) / len(y_coords)
                else:
                    center_x, center_y = 0, 0

                # Filter by confidence score
                if text and text.strip() and score > 0.3:
                    all_detections.append({
                        'text': text.strip(),
                        'center_x': center_x,
                        'center_y': center_y
                    })

        # Build number-to-text mappings
        number_text_map = []
        processed = set()

        # Pattern 1: Embedded numbers (e.g., "1Settings", "2File")
        for i, det in enumerate(all_detections):
            match = re.match(r'^(\d+)([a-zA-Z].*)', det['text'])
            if match:
                number_text_map.append({
                    'number': extract_ascii_digits(match.group(1)),
                    'text': match.group(2),
                    'center_x': det['center_x'],
                    'center_y': det['center_y']
                })
                processed.add(i)

        # Pattern 2: Distance-based matching for separate number/text elements
        numbers = [det for i, det in enumerate(all_detections)
                   if i not in processed and det['text'].isdigit()]
        texts = [det for i, det in enumerate(all_detections)
                 if i not in processed and not det['text'][0].isdigit()]

        for num_det in numbers:
            closest_text = None
            closest_text_det = None
            min_dist = 99999

            # Find nearest text element within 500 pixel radius
            for text_det in texts:
                dist = ((num_det['center_x'] - text_det['center_x']) ** 2 +
                        (num_det['center_y'] - text_det['center_y']) ** 2) ** 0.5
                if dist < 500 and dist < min_dist:
                    min_dist = dist
                    closest_text = text_det['text']
                    closest_text_det = text_det

            if closest_text and closest_text_det:
                number_text_map.append({
                    'number': extract_ascii_digits(num_det['text']),
                    'text': closest_text,
                    'center_x': closest_text_det['center_x'],
                    'center_y': closest_text_det['center_y']
                })

        # Sort mappings by number for consistent ordering
        number_text_map.sort(key=lambda x: int(x['number']) if x['number'].isdigit() else 999)

        # Save OCR mappings to JSON
        json_path = screenshot_path.replace('.png', '_ocr.json')

        with open(json_path, 'w') as f:
            json.dump({'mappings': number_text_map}, f, indent=2)

        # Save metadata pointing to latest OCR JSON
        with open(meta_json_path, 'w') as f:
            json.dump({'json_path': json_path}, f, indent=2)

        print(f"Found {len(number_text_map)} mappings")
        logger.info(f"Found {len(number_text_map)} mappings")

        return {
            'mappings': number_text_map,
            'screenshot_path': screenshot_path,
            'json_path': json_path,
            'status': 'completed'
        }

    
    @mcp_agent_app.workflow_run
    async def run(self) -> WorkflowResult[dict]:
        """
        Execute the complete screen capture and OCR workflow.
        
        Orchestrates the workflow with appropriate timing delays:
        1. Announce "start listening" and wait 5 seconds
        2. Announce "show numbers" and wait 5 seconds for UI update
        3. Capture and analyze the screen
        4. Announce "stop listening"
        
        Returns:
            WorkflowResult[dict]: Result with OCR mappings and status.
        """
        # Prepare for voice listening
        step0 = await self.prepare_screen("start listening")
        time.sleep(5)
        
        # Trigger numbered labels on screen
        step1 = await self.prepare_screen("show numbers")
        time.sleep(5)
        
        # Capture and analyze
        step2 = await self.analyze_screen()

        # End listening mode
        step3 = await self.prepare_screen("stop listening")        

        return WorkflowResult(value={
            'workflow': 'capture_screen_with_numbers',            
            'steps': [step2],
            'status': 'completed',            
        })


@mcp_server.tool()
async def capture_screen_with_numbers_tool():
    """
    MCP tool endpoint for capturing screen with numbered labels and OCR.
    
    Executes the CaptureScreenWithNumbers workflow which:
    1. Triggers display of numbered labels on UI elements
    2. Captures a screenshot
    3. Runs OCR to detect text and numbers
    4. Creates mappings between numbers and UI elements
    5. Saves mappings for use by click_workflow_tool
    
    Returns:
        str: JSON-formatted result of the workflow execution.
    """
    try:        
        workflow1 = CaptureScreenWithNumbers()
        result = await workflow1.run()
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return json.dumps({
            'status': 'error',
            'error': str(e)
        })


# =============================================================================
# DIRECT VOICE COMMAND TOOL
# =============================================================================

@mcp_server.tool()
async def echo_tool(command: str) -> str:
    """
    Direct voice command echo and TTS tool.
    
    Wraps a command in the special voice command format and speaks it aloud.
    Used for simple voice commands that don't require workflow orchestration.
    
    Args:
        command: The voice command text to echo and speak.
        
    Returns:
        str: The command wrapped in ##VC##$...##VC## format.
    """
    result = f"##VC##${command}##VC##"
    speak_text(command)
    return result


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Start the hybrid MCP server.
    
    Launches the FastMCP server with streamable HTTP transport.
    The server listens on port 3000 (configured above) and provides:
    - Workflow-based tools via mcp-agent orchestration
    - HTTP transport via FastMCP streamable-http
    
    Access the MCP endpoint at: http://127.0.0.1:3000/mcp
    """
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