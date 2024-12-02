import gymnasium
import miniwob
import numpy as np
from gymnasium import spaces
import time
from text_similarity_starter_code import compare_strings
from miniwob.action import ActionTypes
import matplotlib.pyplot as plt

import cv2
import numpy as np
import easyocr

import cv2
import numpy as np
import easyocr
import pytesseract
from PIL import Image


class ImageDetector:
    def __init__(self):
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['en'])

    def preprocess_image(self, image_array):
        """
        Enhance image for better OCR
        """
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array.copy()

        # Apply different preprocessing techniques
        preprocessed_images = []

        # Original grayscale
        preprocessed_images.append(gray)

        # Thresholded
        _, thresh1 = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        preprocessed_images.append(thresh1)

        # Otsu's thresholding
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        preprocessed_images.append(thresh2)

        # Adaptive thresholding
        thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
        preprocessed_images.append(thresh3)

        # Add padding
        for i in range(len(preprocessed_images)):
            preprocessed_images[i] = cv2.copyMakeBorder(
                preprocessed_images[i], 10, 10, 10, 10,
                cv2.BORDER_CONSTANT, value=255
            )

        return preprocessed_images

    def detect_text_tesseract(self, image):
        """
        Detect text using Tesseract with custom configuration
        """
        # Configure Tesseract parameters for single character/number detection
        custom_config = r'--oem 3 --psm 10'  # PSM 10 treats image as single character

        try:
            text = pytesseract.image_to_string(image, config=custom_config).strip()
            confidence = 0

            if text:
                # Get confidence scores
                data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
                confidences = [float(conf) for conf in data['conf'] if conf != '-1']
                if confidences:
                    confidence = max(confidences) / 100

                return text, confidence
        except:
            pass

        return None, 0

    def detect_text_easyocr(self, image):
        """
        Detect text using EasyOCR
        """
        try:
            results = self.reader.readtext(image)
            if results:
                text = results[0][1]
                confidence = results[0][2]
                return text, confidence
        except:
            pass

        return None, 0

    def detect_content(self, image_array):
        """
        Detect text/numbers using multiple OCR approaches and preprocessing
        """
        preprocessed_images = self.preprocess_image(image_array)

        best_result = {'type': 'unknown', 'confidence': 0}

        # Try different preprocessing and OCR combinations
        for img in preprocessed_images:
            # Try Tesseract
            text, confidence = self.detect_text_tesseract(img)
            if text and confidence > best_result['confidence']:
                best_result = self._classify_text(text, confidence)

            # Try EasyOCR
            text, confidence = self.detect_text_easyocr(img)
            if text and confidence > best_result['confidence']:
                best_result = self._classify_text(text, confidence)

            # If we found a high-confidence result, stop searching
            if best_result['confidence'] > 0.9:
                break

        # If no text is found or confidence is very low, try shape detection
        if best_result['type'] == 'unknown' or best_result['confidence'] < 0.8:
            shape_result = self.detect_shape(image_array)
            return shape_result

        return best_result

    def _classify_text(self, text, confidence):
        """
        Classify detected text as number or letter
        """
        # Clean the text
        text = text.strip().upper()
        if not text:
            return {'type': 'unknown', 'confidence': 0}

        # Take only the first character if multiple were detected
        text = text[0]

        if text.isdigit():
            return {
                'type': 'number',
                'value': text,
                'confidence': confidence
            }
        elif text.isalpha():
            return {
                'type': 'letter',
                'value': text,
                'confidence': confidence
            }
        return {'type': 'unknown', 'confidence': confidence}

    def detect_shape(self, image_array):
        """
        Detect shapes if no text is found
        """
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array.copy()

        # Threshold the image
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {'type': 'no shape detected'}

        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)

        # Calculate shape properties
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        vertices = len(approx)
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        shape_type = self._classify_shape(vertices, aspect_ratio, solidity)

        return {
            'type': shape_type,
            'properties': {
                'vertices': vertices,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'center': (x + w // 2, y + h // 2)
            }
        }

    def _classify_shape(self, vertices, aspect_ratio, solidity):
        """Helper function to classify shapes"""
        if vertices == 3:
            return "triangle"
        elif vertices == 4:
            if 0.95 <= aspect_ratio <= 1.05:
                return "square"
            else:
                return "rectangle"
        elif vertices > 4:
            if solidity > 0.9:
                return "circle"
            else:
                return "complex shape"
        return "unknown shape"


detector = ImageDetector()


# This is our policy
class GradualMovePolicy:
    def __init__(self, step_size=5):
        self.origin = np.array([0, 0])
        self.step_number = 0

    def __call__(self, observation_):
        if self.step_number == 0:
            shape = detector.detect_content(observation['screenshot'][50:125, 40:115])
            print(shape)

            for element in observation_['dom_elements']:
                if element['text'].lower() == shape['type'].lower():
                    action = env.unwrapped.create_action(
                        ActionTypes.CLICK_ELEMENT,
                        ref=element["ref"]
                    )
                    time.sleep(1.0)
                    return action

            self.step_number += 1

        return {
            "action_type": 1,
            "coords": np.array([0, 0]),
            "key": 0
        }


# This is the main entry
# env = gymnasium.make('miniwob/click-test-2-v1', render_mode='human')
# env = gymnasium.make('miniwob/click-test-2-v1', render_mode='human')
# circle-center
env = gymnasium.make('miniwob/identify-shape', render_mode='human')
# env.unwrapped.instance = env.unwrapped._hard_reset_instance()
env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=99999999999)

# Create our modified environment
# env = TimeBasedRewardWrapper(env)

try:
    # use our policy
    policy = GradualMovePolicy(step_size=5)
    observation, info = env.reset(seed=400)
    print(observation['dom_elements'])

    # show the target
    # assert observation["utterance"] == "Click button ONE."
    # assert observation["fields"] == (("target", "ONE"),)

    print(observation["utterance"], observation["fields"])
    
    final_reward = 0

    # run for some time
    for i in range(10000):
        action = policy(observation)
        observation, reward, terminated, truncated, _ = env.step(action)

        # print(f"Step {i}: Reward={reward}, Position={action['coords']}")

        if terminated or truncated:
            print(f"Episode finished with final reward: {reward}")
            time.sleep(3)
            policy.target_pos = None
            policy.current_pos = None
            policy = GradualMovePolicy(step_size=5)
            observation, info = env.reset()
            print(observation["utterance"], observation["fields"])
finally:
    env.close()