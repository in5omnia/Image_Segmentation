import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import torch
import torchvision.transforms.functional as TF
import base64
import io
import logging

from prompt_model import PromptModel
from clipunet import ClipUNet
from unet.unet import unet
from autoencoder.autoencoder import SegmentationAutoencoder
from utils import *

# Define the paths to the model checkpoints
PROMPT_MODEL_PATH = "../models/prompt.pytorch"
CLIP_MODEL_PATH = "../models/clip.pytorch"
UNET_MODEL_PATH = "../models/unet.pytorch"
AE_MODEL_PATH = "../models/ae.pytorch"

# Set the device to use for computation (CPU or GPU)
DEVICE = torch.device("cpu")
# Define the interpolation method for resizing images
INTERPOLATION = 'bilinear'
# Define the target size for resizing images
TARGET_SIZE = 224
# Configure basic logging settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize a dictionary to store the loaded models
models = {}

def load_model(model_cls, path, model_name, **kwargs):
    """
    Loads a model from a checkpoint file or initializes a new model.

    Args:
        model_cls: The class of the model to load.
        path: The path to the model checkpoint file.
        model_name: A name for the model, used for logging.
        **kwargs: Additional keyword arguments to pass to the model constructor.

    Returns:
        The loaded model, or None if an error occurred.
    """
    try:
        # Initialize different model types with specific parameters based on the model class
        if model_cls == unet:
            model_instance = model_cls(din=3, dout=4, **kwargs).to(DEVICE)
        elif model_cls == SegmentationAutoencoder:
             model_instance = model_cls(din=3, num_classes=4, **kwargs).to(DEVICE)
        elif model_cls == ClipUNet:
             model_instance = model_cls(num_classes=4, **kwargs).to(DEVICE)
        elif model_cls == PromptModel:
             model_instance = model_cls(path=None, **kwargs).to(DEVICE)
        else:
             model_instance = model_cls(**kwargs).to(DEVICE)

        # Load the model checkpoint if it exists
        if os.path.exists(path):
            logger.info(f"Loading checkpoint for {model_name} from {path}...")
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                 state_dict = checkpoint["state_dict"]
            else:
                 state_dict = checkpoint

            # Remove 'module.' prefix if present (common in DataParallel models)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            model_instance.load_state_dict(state_dict)
            logger.info(f"{model_name} model successfully loaded from checkpoint.")
        else:
            logger.warning(f"Checkpoint file '{path}' not found for {model_name}. Using untrained model.")

        model_instance.eval()
        return model_instance
    except Exception as e:
        logger.exception(f"Error loading {model_name} model from {path}: {e}")
        return None

# Load all the models.  The `load_model` function handles loading from checkpoints or initializing untrained models.
models["prompt_model"] = load_model(PromptModel, PROMPT_MODEL_PATH, "Prompt")
models["clip"] = load_model(ClipUNet, CLIP_MODEL_PATH, "ClipUNet")
models["unet"] = load_model(unet, UNET_MODEL_PATH, "UNet")
models["autoencoder"] = load_model(SegmentationAutoencoder, AE_MODEL_PATH, "Autoencoder", pretrained_encoder_path=None)
logger.info(f"Loaded models: {list(models.keys())}")

def decode_base64_image(base64_string):
    """
    Decodes a base64 encoded image string to a PIL Image object.

    Args:
        base64_string: The base64 encoded image string.

    Returns:
        A PIL Image object.
    """
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode == 'RGBA':
        logger.info("Converting RGBA image to RGB image")
        image = image.convert("RGB")
    return image

def encode_pil_to_base64(pil_image, format="PNG"):
    """
    Encodes a PIL Image object to a base64 encoded string.

    Args:
        pil_image: The PIL Image object.
        format: The image format (e.g., "PNG", "JPEG").

    Returns:
        A base64 encoded image string.
    """
    buffered = io.BytesIO()
    pil_image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"

def create_prompt_mask(image_size, prompt_type, prompt_data):
    """
    Creates a mask based on the prompt type and prompt data.

    Args:
        image_size: The size of the image (width, height).
        prompt_type: The type of the prompt ("points", "bbox", "scribble", "text").
        prompt_data: The prompt data.

    Returns:
        A PIL Image object representing the mask.
    """
    width, height = image_size
    mask = Image.new('L', (width, height), 0)
    try:
        if prompt_type == "points":
            radius = 20
            if not prompt_data: return mask
            draw = ImageDraw.Draw(mask)
            if isinstance(prompt_data, dict):
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
            scribble_img = decode_base64_image(prompt_data)
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

# Define a color map for the segmentation classes
COLOR_MAP = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255)
}

# Define class names for the segmentation classes
CLASS_NAMES = {
    "standard": {
        0: "Background",
        1: "Cat",
        2: "Dog",
        3: "Boundary"
    },
    "prompt_model": {
        0: "Deactivated",
        1: "Background+Boundary",
        2: "Cat",
        3: "Dog"
    }
}

# Initialize the Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def index():
    """
    Handles the root route ("/") and renders the index.html template.
    """
    return render_template('index.html')

@app.route('/segment', methods=['POST'])
def segment_image():
    """
    Handles the /segment route (POST method) to perform image segmentation.
    """
    # Ensure the request is a JSON request
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    required_fields = ['image_b64', 'model_type', 'original_width', 'original_height']
    if not all(field in data for field in required_fields):
        missing = [field for field in required_fields if field not in data]
        logger.warning(f"Missing required fields: {missing}. Received keys: {list(data.keys())}")
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    model_type = data['model_type']
    if model_type not in models or models[model_type] is None:
        logger.error(f"Requested model type '{model_type}' is not available or failed to load.")
        return jsonify({"error": f"Model type '{model_type}' not available"}), 400

    selected_model = models[model_type]
    logger.info(f"Using model type: {model_type}")

    if model_type == "prompt_model" and 'prompt_data' not in data:
         logger.warning("Missing 'prompt_data' for prompt_model type.")
         return jsonify({"error": "Missing 'prompt_data' field required for prompt_model"}), 400

    uploaded_label_b64 = data.get('label_b64', None)

    try:
        original_image = decode_base64_image(data['image_b64'])
        logger.info(f"Received image of size: {original_image.size}")

        prompt_type = data.get('prompt_type', 'points')
        prompt_data = data.get('prompt_data', None)
        original_size = (data['original_width'], data['original_height'])

        prompt_mask = None
        prompt_tensor = None
        resized_prompt_input = None
        if model_type == "prompt_model":
            # Create the prompt mask based on the prompt type and data
            prompt_mask = create_prompt_mask(original_size, prompt_type, prompt_data)
            if prompt_mask.size != original_image.size:
                 logger.warning(f"Resizing prompt mask from {prompt_mask.size} to {original_image.size}")
                 prompt_mask = prompt_mask.resize(original_image.size, Image.NEAREST)
            prompt_tensor = TF.to_tensor(prompt_mask).unsqueeze(0).to(DEVICE)
            resized_prompt_input, _ = process_batch_forward(prompt_tensor, target_size=TARGET_SIZE)
            resized_prompt_input = resized_prompt_input.to(DEVICE)
            logger.info(f"Generated prompt mask and tensor for prompt_model.")
        else:
            logger.info(f"Skipping prompt mask generation for model type: {model_type}")

        img_tensor = TF.to_tensor(original_image).unsqueeze(0).to(DEVICE)
        img_tensor = img_tensor.to(DEVICE)

        with torch.no_grad():
            resized_model_input, meta_list = process_batch_forward(img_tensor, target_size=TARGET_SIZE)
            resized_model_input = resized_model_input.to(DEVICE)
            logger.info(f"Resized model input shape: {resized_model_input.shape}")

            if model_type == "prompt_model":
                if resized_prompt_input is None:
                     logger.error("Prompt input tensor is missing for prompt_model inference.")
                     return jsonify({"error": "Internal error: Prompt input missing for prompt_model"}), 500
                logger.info(f"Resized prompt input shape: {resized_prompt_input.shape}")
                model_output = selected_model(resized_model_input, resized_prompt_input)
            else:
                model_output = selected_model(resized_model_input)


            model_output = process_batch_reverse(model_output, meta_list, interpolation=INTERPOLATION)
            arr = np.array(model_output)
            model_output = model_output[0].argmax(0)
            
            model_output = model_output.to(torch.uint8)
            unique_classes = torch.unique(model_output).cpu().numpy()
            logger.info(f"DEBUG Unique predicted class indices: {unique_classes}")

        predicted_mask_tensor = None
        if isinstance(model_output, torch.Tensor):
            predicted_mask_tensor = model_output
            logger.info("Model returned a single tensor.")

        elif isinstance(model_output, (list, tuple)) and len(model_output) == 1 and isinstance(model_output[0], torch.Tensor):
            predicted_mask_tensor = model_output[0]
            logger.info("Model returned a list/tuple with one tensor.")
        else:
             logger.error(f"Unexpected model output type or structure: {type(model_output)}")
             raise TypeError(f"Model output was not a single tensor or a list/tuple containing one tensor. Got: {type(model_output)}")

        predicted_mask_pil = None
        try:
            mask_np = predicted_mask_tensor.cpu().numpy()

            if mask_np.dtype != np.uint8:
                 mask_np = mask_np.astype(np.uint8)

            mask_np[mask_np == 255] = 3
            logger.info(f"Remapped mask values 255 to 3. Unique values now: {np.unique(mask_np)}")

            height, width = mask_np.shape
            color_mask_np = np.zeros((height, width, 3), dtype=np.uint8)
            for class_index, color in COLOR_MAP.items():
                color_mask_np[mask_np == class_index] = color

            predicted_mask_pil = Image.fromarray(color_mask_np, 'RGB')
            logger.info(f"Converted prediction indices to COLOR PIL image (mode: {predicted_mask_pil.mode})")

        except TypeError as te:
             logger.exception(f"TypeError converting tensor to PIL: {te}")
             return jsonify({"error": f"Internal error converting prediction: {te}"}), 500
        except Exception as pil_e:
             logger.exception(f"Error during PIL conversion or saving output mask: {pil_e}")
             return jsonify({"error": f"Internal error processing output mask: {pil_e}"}), 500

        predicted_mask_b64 = encode_pil_to_base64(predicted_mask_pil)

        colored_label_b64 = None
        if uploaded_label_b64:
            try:
                logger.info("Processing uploaded label mask...")
                uploaded_label_pil = decode_base64_image(uploaded_label_b64)
                logger.info(f"DEBUG: Initial mode of uploaded label PIL image: {uploaded_label_pil.mode}")

                if uploaded_label_pil.mode in ('L', 'P'):
                    label_np = np.array(uploaded_label_pil)
                    logger.info(f"DEBUG: Reading label as mode '{uploaded_label_pil.mode}'. Unique values in NumPy array: {np.unique(label_np)}")
                elif uploaded_label_pil.mode in ('RGB', 'RGBA'):
                    logger.warning(f"Uploaded label is mode '{uploaded_label_pil.mode}'. Attempting conversion to 'L' for indices, but this may be inaccurate. Expected 'L' or 'P' mode index mask.")
                    uploaded_label_pil_l = uploaded_label_pil.convert('L')
                    label_np = np.array(uploaded_label_pil_l)
                    logger.info(f"DEBUG: Converted '{uploaded_label_pil.mode}' to 'L'. Unique values in NumPy array: {np.unique(label_np)}")
                else:
                    logger.error(f"Unsupported mode '{uploaded_label_pil.mode}' for uploaded label mask. Cannot process.")
                    raise ValueError(f"Unsupported label mask mode: {uploaded_label_pil.mode}")

                if label_np.dtype != np.uint8:
                     label_np = label_np.astype(np.uint8)
                     logger.info(f"DEBUG: Converted label array dtype to uint8.")

                original_unique_values = np.unique(label_np)
                label_np[label_np == 255] = 3
                remapped_unique_values = np.unique(label_np)
                if 255 in original_unique_values:
                     logger.info(f"Remapped uploaded label values 255 to 3. Unique values now: {remapped_unique_values}")
                else:
                     logger.info(f"No value 255 found to remap. Unique values remain: {remapped_unique_values}")

                label_height, label_width = label_np.shape
                color_label_np = np.zeros((label_height, label_width, 3), dtype=np.uint8)
                for class_index, color in COLOR_MAP.items():
                    if class_index in remapped_unique_values:
                         color_label_np[label_np == class_index] = color

                unique_colors_in_array = np.unique(color_label_np.reshape(-1, 3), axis=0)
                logger.info(f"DEBUG: Unique RGB colors applied to label array: {unique_colors_in_array.tolist()}")

                colored_label_pil = Image.fromarray(color_label_np, 'RGB')
                logger.info(f"DEBUG: Mode of final colored label PIL image: {colored_label_pil.mode}")
                colored_label_b64 = encode_pil_to_base64(colored_label_pil)
                logger.info("Successfully processed and color-mapped uploaded label.")

            except Exception as label_proc_e:
                logger.exception(f"Error processing uploaded label mask: {label_proc_e}. Returning original label if available.")
                logger.error(f"Error processing uploaded label mask: {label_proc_e}. Returning original label if available.")
                colored_label_b64 = uploaded_label_b64

        logger.info("Segmentation successful.")
        class_names_for_model = CLASS_NAMES.get(model_type, CLASS_NAMES["standard"])

        return jsonify({
            "output_label_b64": colored_label_b64,
            "output_mask_b64": predicted_mask_b64,
            "message": "Segmentation successful.",
            "model_type": model_type,
            "class_names": class_names_for_model
        })

    except Exception as e:
        logger.exception(f"Error during segmentation processing: {e}")
        return jsonify({"error": f"An internal error occurred: {e}"}), 500

if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True, host='127.0.0.1', port=5000)