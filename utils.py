import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms.functional
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from tqdm import tqdm

def plot_tensor_with_custom_colors(tensor, color_map):
    """
    Plots a tensor with custom colors using matplotlib.

    Args:
        tensor (torch.Tensor): A 2D or 3D tensor with integer values representing different categories.
                                The values should correspond to keys in the color_map.
        color_map (dict): A dictionary mapping integer values to RGB color tuples (e.g., (0, 0, 0) for black).
    """

    # Convert to numpy if it's a tensor
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()

    # Ensure the tensor is 2D or 3D
    if len(tensor.shape) not in [2, 3]:
        raise ValueError("Tensor must be 2D or 3D")

    # Create an RGB image from the color map
    height, width = tensor.shape[:2]  # Handle both 2D and 3D tensors
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    for value, color in color_map.items():
        mask = (tensor == value)
        rgb_image[mask] = color

    # Display the image using matplotlib
    plt.imshow(rgb_image)
    plt.show()

# Define the color map
color_map = {
    0: (0, 0, 0),      # Black
    1: (255, 0, 0),    # Red
    2: (0, 255, 0),    # Green
    3: (255, 255, 255),  # White
    255: (255,255,255)
}

# Assume these are defined somewhere in your code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = False  # or False depending on your setup

def resize_with_padding(image, target_size=512):
    """
    Resize a single image (Tensor of shape (C, H, W)) so that the longer side
    equals target_size, preserving aspect ratio; add black padding as needed.
    Returns the resized and padded image, plus a metadata dictionary.
    """
    _, orig_h, orig_w = image.shape
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    
    # Resize the image
    image_resized = TF.resize(image, size=(new_h, new_w))
    
    # Compute padding on each side
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # Pad the image (padding order: left, top, right, bottom)
    image_padded = TF.pad(image_resized, padding=(pad_left, pad_top, pad_right, pad_bottom), fill=0)

    meta = {
        "original_size": (orig_h, orig_w),
        "new_size": (new_h, new_w),
        "pad": (pad_left, pad_top, pad_right, pad_bottom),
        "scale": scale
    }
    return image_padded, meta

def reverse_resize_and_padding(image, meta, interpolation="bilinear"):
    """
    Remove the padding from image (Tensor of shape (C, target_size, target_size))
    using metadata and then resize the cropped image back to the original size.
    interpolation: "bilinear" for continuous outputs; use "nearest" for label maps.
    """
    pad_left, pad_top, pad_right, pad_bottom = meta["pad"]
    new_h, new_w = meta["new_size"]
    
    # Crop out the padding: from pad_top to pad_top+new_h and pad_left to pad_left+new_w.
    image_cropped = image[..., pad_top: pad_top + new_h, pad_left: pad_left + new_w]
    
    # Resize the cropped image back to the original size.
    orig_h, orig_w = meta["original_size"]
    # F.interpolate expects a 4D tensor.
    image_original = F.interpolate(image_cropped.unsqueeze(0),
                                   size=(orig_h, orig_w),
                                   mode=interpolation,
                                   align_corners=False if interpolation != "nearest" else None)
    return image_original.squeeze(0)

def process_batch_forward(batch_images, target_size=512):
    """
    Process a batch (Tensor of shape (N, C, H, W)) by resizing each image to target_size
    with aspect ratio preserved (adding black padding).
    Returns the processed batch and a list of meta dictionaries.
    """
    resized_batch = []
    meta_list = []
    for image in batch_images:
        image_resized, meta = resize_with_padding(image, target_size)
        resized_batch.append(image_resized)
        meta_list.append(meta)
    return torch.stack(resized_batch), meta_list

def process_batch_reverse(batch_outputs, meta_list, interpolation="bilinear"):
    """
    Given a batch of network outputs of shape (N, C, target_size, target_size) and the
    corresponding meta info, reverse the transform for each one to obtain predictions at their
    original sizes.
    """
    original_outputs = []
    for output, meta in zip(batch_outputs, meta_list):
        restored = reverse_resize_and_padding(output, meta, interpolation=interpolation)
        original_outputs.append(restored)
    return original_outputs
