import os
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import torch
import torchvision.transforms.functional as TF
import base64
import io
import logging

from prompt_model import SegmentationAutoencoder
from utils import *


# --- Configuration ---
# Load environment variables if using .env
# from dotenv import load_dotenv
# load_dotenv()
MODEL_PATH ="autoencoder_256_with_aug_noAugEnc_weight2_ce_dice_adamw_64.pytorch" # Default model path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INTERPOLATION = 'bilinear'
TARGET_SIZE = 256

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Placeholder for your Segmentation Model ---
# Replace this with loading and using your actual trained model


# --- Load Model (Load once on startup) ---
try:
    model = SegmentationAutoencoder(3, num_classes=4).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        # Load state dict if your Network class matches the saved structure
        # Load the checkpoint dictionary; move tensors to the correct device
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])

        logger.info(f"Placeholder model initialized. Replace loading logic if using '{MODEL_PATH}'.")
    else:
        logger.warning(f"Model file '{MODEL_PATH}' not found. Using untrained placeholder model.")
    model.eval() # Set to evaluation mode
except Exception as e:
    logger.error(f"Error loading model: {e}")
    # Handle error appropriately, maybe exit or use a dummy model
    model = None

# --- Helper Functions (Backend Image Processing) ---

def decode_base64_image(base64_string):
    """Decodes a base64 string into a PIL Image."""
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))
    return image.convert("RGB") # Ensure RGB format

def encode_pil_to_base64(pil_image, format="PNG"):
    """Encodes a PIL image into a base64 string."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"

def create_prompt_mask(image_size, prompt_type, prompt_data):
    """Creates a single-channel prompt mask (heatmap) based on prompt type."""
    width, height = image_size
    mask = Image.new('L', (width, height), 0) # Grayscale mask

    try:
        if prompt_type == "points":
            radius = 20 # Adjust heatmap point radius
            if not prompt_data: return mask # Empty mask if no points
            draw = ImageDraw.Draw(mask)
            for point in prompt_data: # List of {'x': float, 'y': float}
                x, y = int(point['x']), int(point['y'])
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
            mask = mask.filter(ImageFilter.GaussianBlur(radius=radius // 2))
            logger.info(f"Generated heatmap from {len(prompt_data)} points.")

        elif prompt_type == "bbox":
             # prompt_data expected: {'x': xmin, 'y': ymin, 'width': w, 'height': h}
             if not prompt_data: return mask
             x, y, w, h = int(prompt_data['x']), int(prompt_data['y']), int(prompt_data['width']), int(prompt_data['height'])
             if w > 0 and h > 0:
                 draw = ImageDraw.Draw(mask)
                 draw.rectangle([x, y, x + w, y + h], fill=255)
                 logger.info(f"Generated mask from bbox: [{x},{y},{x+w},{y+h}].")

        elif prompt_type == "scribble":
            # prompt_data expected: base64 encoded image of the scribble overlay
            if not prompt_data: return mask
            scribble_overlay = decode_base64_image(prompt_data).convert('L') # Decode and ensure grayscale
            if scribble_overlay.size != image_size:
                # Resize scribble if necessary (should ideally match on frontend)
                scribble_overlay = scribble_overlay.resize(image_size, Image.NEAREST)
            # Convert scribble to binary mask (assuming black background, non-black scribble)
            mask_array = np.array(scribble_overlay)
            threshold = 10 # Consider anything not near black as part of the scribble
            binary_mask_array = np.where(mask_array > threshold, 255, 0).astype(np.uint8)
            mask = Image.fromarray(binary_mask_array, mode='L')
            logger.info("Generated mask from scribble.")

        elif prompt_type == "text":
            # Model specific: How to convert text to a spatial mask?
            # Option 1: Ignore text for spatial mask (use empty mask)
            # Option 2: Use a separate model (like CLIP) to generate a heatmap (complex)
            # Option 3: Your model takes text differently (modify model input logic)
            logger.warning("Text prompt received, generating empty spatial mask for this example.")
            # mask remains empty / black
    except Exception as e:
        logger.error(f"Error creating prompt mask for type {prompt_type}: {e}")
        # Return empty mask on error
        mask = Image.new('L', (width, height), 0)

    return mask

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='templates', static_folder='static')

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/segment', methods=['POST'])
def segment_image():
    """Handles the segmentation request."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    required_fields = ['image_b64', 'prompt_type', 'prompt_data', 'original_width', 'original_height']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields in request"}), 400

    try:
        # 1. Decode Input Image
        original_image_b64 = data['image_b64']
        original_image = decode_base64_image(original_image_b64)
        logger.info(f"Received image of size: {original_image.size}")

        # 2. Generate Prompt Mask (Heatmap)
        prompt_type = data['prompt_type']
        prompt_data = data['prompt_data']
        # Use original dimensions reported by frontend for mask generation
        original_size = (data['original_width'], data['original_height'])
        prompt_mask = create_prompt_mask(original_size, prompt_type, prompt_data)

        # Ensure prompt mask has the same size as the input image if needed (resize maybe?)
        if prompt_mask.size != original_image.size:
             logger.warning(f"Resizing prompt mask from {prompt_mask.size} to {original_image.size}")
             prompt_mask = prompt_mask.resize(original_image.size, Image.NEAREST)


        # 3. Preprocess for Model
        # Normalize based on your model's training
        # Example: Simple scaling to [0, 1]
        img_tensor = TF.to_tensor(original_image)      # C, H, W
        prompt_tensor = TF.to_tensor(prompt_mask)     # 1, H, W

        # Add batch dimension and move to device
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE) # B, C, H, W
        prompt_tensor = prompt_tensor.unsqueeze(0).to(DEVICE) # B, 1, H, W

        # Ensure spatial dimensions match (should be guaranteed by resize above)
        assert img_tensor.shape[2:] == prompt_tensor.shape[2:]

        # Concatenate image and prompt mask
        # Input channels = image channels (3) + prompt channels (1)
        model_input = torch.cat([img_tensor, prompt_tensor], dim=1) # B, 4, H, W


        # 4. Run Model Inference
        with torch.no_grad():

            resized_model_input, meta_list = process_batch_forward(model_input, target_size=TARGET_SIZE)   # resize X for network
            resized_model_input = resized_model_input.to(DEVICE)
                        
                        # Compute prediction
            output1_tensor = model(resized_model_input)
            output1_tensor = process_batch_reverse(output1_tensor, meta_list, interpolation=INTERPOLATION)
            
            #output1_tensor = model(model_input) # Expect B, 1, H, W

        #if label.shape[1] == 4 and label.ndim == 4:  
                #print(f"    Converting original image from RGBA to RGB")
                #label = label[:, :3, :, :] # Keep only the first 3 channels (R, G, B)
        #try:
        #    label = getLabel()
        #except:
        label = img_tensor

        # 5. Postprocess Output Tensors
        # Remove batch dim, move to CPU, convert to PIL
        # Assumes output tensors are in [0, 1] range (due to Sigmoid)
        output1_pil = TF.to_pil_image(output1_tensor.squeeze(0).cpu())
        output2_pil = TF.to_pil_image(label.squeeze(0).cpu())

        # 6. Encode Output Images to Base64
        output1_b64 = encode_pil_to_base64(output1_pil)
        output2_b64 = encode_pil_to_base64(output2_pil)

        logger.info("Segmentation successful.")
        return jsonify({
            "output_label_b64": output1_b64,
            "output_mask_b64": output2_b64,
            "message": "Segmentation successful."
        })

    except Exception as e:
        logger.exception(f"Error during segmentation processing: {e}") # Log full traceback
        return jsonify({"error": f"An internal error occurred: {e}"}), 500


# --- Run Flask App ---
if __name__ == '__main__':
    # Set host='0.0.0.0' to make it accessible on your network (use with caution)
    app.run(debug=True, host='127.0.0.1', port=5000) # debug=True for development