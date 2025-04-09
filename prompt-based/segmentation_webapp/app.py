import os
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import torch
import torchvision.transforms.functional as TF
import base64
import io
import logging


from prompt_model import PromptModel #SegmentationAutoencoder
from utils import *

# --- Configuration ---
MODEL_PATH ="/Users/beatrizgavilan/Desktop/Assignments/CV/Image_Segmentation/prompt-based/prompt_256_ce_dice_weight_full_forwebapp.pytorch" 
#MODEL_PATH = "/Users/beatrizgavilan/Desktop/Assignments/CV/Image_Segmentation/prompt-based/autoencoderSeg_256_withAug_ce_dice_weight2.pytorch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INTERPOLATION = 'bilinear'
TARGET_SIZE = 224
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# --- Load Model ---
try:
    model = PromptModel(path=None).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        # Load state dict if your Network class matches the saved structure
        # Load the checkpoint dictionary; move tensors to the correct device
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        # Load model state

        model.load_state_dict(checkpoint["model_state_dict"])

        logger.info(f"Model succesfully initialized.")
    else:
        logger.warning(f"Model file '{MODEL_PATH}' not found. Using untrained placeholder model.")
    
    model.eval() # Set to evaluation mode
except Exception as e:
    logger.error(f"Error loading model.")
    model = None

# --- Helper Functions (decode_base64_image, encode_pil_to_base64, create_prompt_mask) ---
# Keep these exactly as they were in the previous version
def decode_base64_image(base64_string):
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))
    # Check for RGBA and convert if necessary (like the log message indicated)
    if image.mode == 'RGBA':
        logger.info("Converting RGBA image to RGB image")
        image = image.convert("RGB")
    return image

def encode_pil_to_base64(pil_image, format="PNG"):
    buffered = io.BytesIO()
    pil_image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"

def create_prompt_mask(image_size, prompt_type, prompt_data):
    width, height = image_size
    mask = Image.new('L', (width, height), 0)
    try:
        if prompt_type == "points":
            radius = 20
            if not prompt_data: return mask
            draw = ImageDraw.Draw(mask)
            # Ensure prompt_data is a list here, even if frontend sent just one point
            if isinstance(prompt_data, dict): # Handle if frontend sent single dict instead of list
                prompt_data = [prompt_data]
            for point in prompt_data:
                if isinstance(point, dict) and 'x' in point and 'y' in point:
                     x, y = int(point['x']), int(point['y'])
                     draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
                else:
                     logger.warning(f"Skipping invalid point data: {point}")
            mask = mask.filter(ImageFilter.GaussianBlur(radius=radius // 2))
            logger.info(f"Generated heatmap from points.")
        elif prompt_type == "bbox":
             if not prompt_data or not all(k in prompt_data for k in ('x','y','width','height')): return mask
             x, y, w, h = int(prompt_data['x']), int(prompt_data['y']), int(prompt_data['width']), int(prompt_data['height'])
             if w > 0 and h > 0:
                 draw = ImageDraw.Draw(mask)
                 draw.rectangle([x, y, x + w, y + h], fill=255)
                 logger.info(f"Generated mask from bbox.")
        elif prompt_type == "scribble":
            if not prompt_data: return mask
            # Ensure scribble decoding handles potential RGBA from canvas.toDataURL
            scribble_img = decode_base64_image(prompt_data) # decode_base64 now handles RGB conversion
            scribble_overlay = scribble_img.convert('L')
            if scribble_overlay.size != image_size:
                scribble_overlay = scribble_overlay.resize(image_size, Image.NEAREST)
            mask_array = np.array(scribble_overlay)
            threshold = 10
            binary_mask_array = np.where(mask_array > threshold, 255, 0).astype(np.uint8)
            mask = Image.fromarray(binary_mask_array, mode='L')
            logger.info("Generated mask from scribble.")
        elif prompt_type == "text":
            logger.warning("Text prompt received, generating empty spatial mask for this example.")
    except Exception as e:
        logger.error(f"Error creating prompt mask for type {prompt_type}: {e}")
        mask = Image.new('L', (width, height), 0)
    return mask
# --- End Helper Functions ---



COLOR_MAP = {
    0: (68.08, 1.24, 84.00),       # Background
    1: (48.61, 103.80, 141.80),     # Class 1 (e.g., Red)
    2: (53.04, 183.26, 120.58),     # Class 2 (e.g., Lime Green)
    3: (253.27, 231.07, 36.70)      # Class 3 (e.g., Blue)
    # Add more entries if NUM_CLASSES > 4
}

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='templates', static_folder='static')

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/segment', methods=['POST'])
def segment_image():
    if model is None: return jsonify({"error": "Model not loaded"}), 500
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    # Check for essential fields
    required_fields = ['image_b64', 'prompt_type', 'prompt_data', 'original_width', 'original_height']
    if not all(field in data for field in required_fields):
        logger.warning(f"Missing fields. ")
        #logger.warning(f"Missing fields. Received: {list(data.keys())}")
        return jsonify({"error": "Missing required fields in request"}), 400

    # Get optional label data
    uploaded_label_b64 = data.get('label_b64', None) # Get label if sent, otherwise None

    try:
        # 1. Decode Input Image
        original_image = decode_base64_image(data['image_b64'])
        logger.info(f"Received image of size: {original_image.size}")

        # 2. Generate Prompt Mask
        prompt_type = data['prompt_type']
        prompt_data = data['prompt_data']
        original_size = (data['original_width'], data['original_height'])
        prompt_mask = create_prompt_mask(original_size, prompt_type, prompt_data)
        if prompt_mask.size != original_image.size:
             logger.warning(f"Resizing prompt mask from {prompt_mask.size} to {original_image.size}")
             prompt_mask = prompt_mask.resize(original_image.size, Image.NEAREST)

        # 3. Preprocess for Model
        img_tensor = TF.to_tensor(original_image).unsqueeze(0).to(DEVICE)
        prompt_tensor = TF.to_tensor(prompt_mask).unsqueeze(0).to(DEVICE)
        assert img_tensor.shape[2:] == prompt_tensor.shape[2:]
        #model_input = torch.cat([img_tensor, prompt_tensor], dim=1)

        # --- SAVE THE INPUT IMAGE ---
        """OUTPUT_SAVE_DIR = "output_masks"
        img_pil = None # Initialize
        try:
            img_pil = TF.to_pil_image(img_tensor.squeeze(0).cpu())
            logger.info(f"Converted prediction tensor to PIL image (mode: {img_pil.mode})")
            try:
                filename = f"img.png"
                save_path = os.path.join(OUTPUT_SAVE_DIR, filename)
                img_pil.save(save_path)
                logger.info(f"Successfully saved input image to '{save_path}'")
            except Exception as save_e:
                # Log error but don't crash the request if saving fails
                logger.error(f"Failed to save input image to to {OUTPUT_SAVE_DIR}: {save_e}")
            # --- END SAVING LOGIC ---
        except TypeError as te:
             logger.exception(f"TypeError converting tensor to PIL: {te}")
             return jsonify({"error": f"Internal error converting prediction: {te}"}), 500
        except Exception as pil_e:
             logger.exception(f"Error during PIL conversion or saving output mask: {pil_e}")
             return jsonify({"error": f"Internal error processing output mask: {pil_e}"}), 500
        # --- END SAVE INPUT IMAGE LOGIC ---"""

        # 4. Run Model Inference - Expecting ONE output tensor
        with torch.no_grad():
            resized_model_input, meta_list = process_batch_forward(img_tensor, target_size=TARGET_SIZE)   # resize X for network
            resized_prompt_input, _ = process_batch_forward(prompt_tensor, target_size=TARGET_SIZE) # resize prompt for network
            resized_model_input, resized_prompt_input = resized_model_input.to(DEVICE), resized_prompt_input.to(DEVICE)

            print("resized_model_input.shape", resized_model_input.shape)
            print("resized_prompt_input.shape", resized_prompt_input.shape)

            # Compute prediction
            model_output = model(resized_model_input, resized_prompt_input)
            logger.info(f"HEREEEE1 Model output shape: {model_output.shape}")

            model_output = process_batch_reverse(model_output, meta_list, interpolation=INTERPOLATION)
            arr = np.array(model_output)
            logger.info(f"HEREEEE Model output shape: {arr.shape}")
            model_output = model_output[0].argmax(0)
            
            model_output = model_output.to(torch.uint8) # or .byte() 
            unique_classes = torch.unique(model_output).cpu().numpy()
            logger.info(f"DEBUG Unique predicted class indices: {unique_classes}")

            

        # --- FIX: Handle single model output (direct tensor or list/tuple with one tensor) ---
        predicted_mask_tensor = None
        if isinstance(model_output, torch.Tensor):
            predicted_mask_tensor = model_output # Direct tensor output
            logger.info("Model returned a single tensor.")

        elif isinstance(model_output, (list, tuple)) and len(model_output) == 1 and isinstance(model_output[0], torch.Tensor):
            predicted_mask_tensor = model_output[0] # Tensor inside a list/tuple
            logger.info("Model returned a list/tuple with one tensor.")
        else:
             logger.error(f"Unexpected model output type or structure: {type(model_output)}")
             raise TypeError(f"Model output was not a single tensor or a list/tuple containing one tensor. Got: {type(model_output)}")
        # --- End FIX ---
        
        # --- SAVE THE OUTPUT MASK ---
        OUTPUT_SAVE_DIR = "output_masks"
        predicted_mask_pil = None # Initialize
        try:
            # Convert indices tensor to numpy array on CPU
            mask_np = predicted_mask_tensor.cpu().numpy().astype(np.uint8)

            # Create an empty RGB image array
            height, width = mask_np.shape
            color_mask_np = np.zeros((height, width, 3), dtype=np.uint8)

            # Apply the color map
            for class_index, color in COLOR_MAP.items():
                color_mask_np[mask_np == class_index] = color

            # Convert the numpy color array to a PIL Image
            predicted_mask_pil = Image.fromarray(color_mask_np, 'RGB') # Use 'RGB' mode
            logger.info(f"Converted prediction indices to COLOR PIL image (mode: {predicted_mask_pil.mode})")

            try:
                filename = f"output_mask.png"
                save_path = os.path.join(OUTPUT_SAVE_DIR, filename)
                predicted_mask_pil.save(save_path)
                logger.info(f"Successfully saved output mask to '{save_path}'")
            except Exception as save_e:
                # Log error but don't crash the request if saving fails
                logger.error(f"Failed to save output mask image to {OUTPUT_SAVE_DIR}: {save_e}")
            # --- END SAVING LOGIC ---
            
        except TypeError as te:
             logger.exception(f"TypeError converting tensor to PIL: {te}")
             return jsonify({"error": f"Internal error converting prediction: {te}"}), 500
        except Exception as pil_e:
             logger.exception(f"Error during PIL conversion or saving output mask: {pil_e}")
             return jsonify({"error": f"Internal error processing output mask: {pil_e}"}), 500
        # --- END SAVE LOGIC ---


        # 5. Postprocess Predicted Mask
        #predicted_mask_pil = TF.to_pil_image(predicted_mask_tensor.cpu())

        # 6. Encode Predicted Mask to Base64
        predicted_mask_b64 = encode_pil_to_base64(predicted_mask_pil)

        logger.info("Segmentation successful.")
        # 7. Return JSON including the predicted mask AND the original label (if provided)
        return jsonify({
            "output_label_b64": uploaded_label_b64, # Pass back the original label B64
            "output_mask_b64": predicted_mask_b64,  # The mask generated by the model
            "message": "Segmentation successful."
        })

    except Exception as e:
        logger.exception(f"Error during segmentation processing: {e}")
        return jsonify({"error": f"An internal error occurred: {e}"}), 500

# --- Run Flask App ---
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)




