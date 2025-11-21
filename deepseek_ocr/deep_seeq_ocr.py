import ollama

# Replace 'llama3.2-vision' with your multimodal model
model_name = 'hf.co/unsloth/Qwen2.5-Omni-3B-GGUF:BF16'
image_path = '/Users/ibk5106/Desktop/test.png'


def load_image_bytes(image_path):
    """Load image as bytes - more reliable than file paths"""
    try:
        with open(image_path, 'rb') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        return None

image_path = '/Users/ibk5106/Desktop/test.png'

# Load image as bytes
image_bytes = load_image_bytes(image_path)

if image_bytes:
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': 'What is happening in this image?',
                    'images': [image_bytes],  # Pass image bytes instead of path
                }
            ]
        )

        print(response['message']['content'])

    except Exception as e:
        print(f"Error: {e}")
else:
    print("Failed to load image")