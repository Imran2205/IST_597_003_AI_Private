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
            text_t, confidence_t = self.detect_text_tesseract(img)
            if text_t and confidence_t > best_result['confidence']:
                best_result = self._classify_text(text_t, confidence_t)

            # Try EasyOCR
            text_e, confidence_e = self.detect_text_easyocr(img)
            if text_e and confidence_e > best_result['confidence']:
                best_result = self._classify_text(text_e, confidence_e)

            if text_t == 'A' and text_e != 'A':
                best_result = {'type': 'unknown', 'confidence': 0}
                break

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

"""
# Before using this code please install opencv-python, easyocr, and pytesseract using the following command
# pip install opencv-python easyocr pytesseract
# Now we can use detector to identify the content from the screenshot of the current window.
# Use the following line to get the type of the shape shown in the screenshot

shape = detector.detect_content(observation['screenshot'][50:125, 40:115]) # Here the observation comes from the miniwob environment

# The shape variable contains a dictionary in the following format:
# {'type': 'number', 'value': '7', 'confidence': 0.96}
# {'type': 'letter', 'value': 'W', 'confidence': 0.9762827002734618}
# {'type': 'triangle', 'properties': {'vertices': 3, 'area': 435.0, 'aspect_ratio': 1.0, 'solidity': 0.9688195991091314, 'center': (37, 34)}}

# Use shape['type'] to get the name of the shape.
# shape['type'] can have any of the following values:  square, rectangle, triangle, circle, letter, or number
"""
