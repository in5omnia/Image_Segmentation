import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm
from PIL import Image
from typing import Optional, List


def resize_with_padding(image, target_size=512, interpolation=InterpolationMode.BILINEAR):
    """
    Resize a single image (Tensor of shape (C, H, W)) so that the longer side
    equals target_size, preserving aspect ratio; add black padding as needed.
    Args:
        image (Tensor): Input image tensor of shape (C, H, W).
        target_size (int): The target size for the longer side of the image.
        interpolation (InterpolationMode): The interpolation method to use.
    Returns:
        tuple: A tuple containing the resized and padded image, plus a metadata dictionary.
    """
    _, orig_h, orig_w = image.shape
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    
    # Resize the image
    image_resized = TF.resize(image, size=(new_h, new_w), interpolation=interpolation)
    
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
    Args:
        image (Tensor): The input image tensor.
        meta (dict): The metadata dictionary containing information about the original and new sizes and padding.
        interpolation (str): The interpolation method to use ("bilinear" or "nearest").
    Returns:
        Tensor: The image with padding removed and resized to the original size.
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

def process_batch_forward(batch_images, target_size=512, interpolation=InterpolationMode.BILINEAR):
    """
    Process a batch (Tensor of shape (N, C, H, W)) by resizing each image to target_size
    with aspect ratio preserved (adding black padding).
    Args:
        batch_images (Tensor): A batch of images (N, C, H, W).
        target_size (int): The target size for the longer side of the image.
        interpolation (InterpolationMode): The interpolation method to use.
    Returns:
        tuple: A tuple containing the processed batch and a list of meta dictionaries.
    """
    resized_batch = []
    meta_list = []
    for image in batch_images:
        # If the image has 4 channels, slice to keep only the first 3 (RGB).
        if image.ndim == 3 and image.shape[0] == 4:
            image = image[:3, ...]
        image_resized, meta = resize_with_padding(image, target_size, interpolation)
        resized_batch.append(image_resized)
        meta_list.append(meta)
    return torch.stack(resized_batch), meta_list

def process_batch_reverse(batch_outputs, meta_list, interpolation="bilinear"):
    """
    Given a batch of network outputs of shape (N, C, target_size, target_size) and the
    corresponding meta info, reverse the transform for each one to obtain predictions at their
    original sizes.
    Args:
        batch_outputs (Tensor): A batch of network outputs (N, C, target_size, target_size).
        meta_list (list): A list of meta dictionaries, one for each image in the batch.
        interpolation (str): The interpolation method to use ("bilinear" or "nearest").
    Returns:
        list: A list of tensors, each with the original size.
    """
    original_outputs = []
    for output, meta in zip(batch_outputs, meta_list):
        restored = reverse_resize_and_padding(output, meta, interpolation=interpolation)
        original_outputs.append(restored)
    return original_outputs

def calculate_class_weights(
    label_source,
    num_classes: int,
    ignore_index: Optional[int] = None,
    source_type: str = 'files',
    unimportant_class_indices: Optional[List[int]] = None, # Indices to down-weight
    target_unimportant_weight: float = 1.0, # Target weight for unimportant classes
    normalize_target_sum: float = -1.0 # Normalize weights sum (-1 means num_classes)
) -> torch.Tensor:
    """
    Calculates class weights based on inverse frequency, then adjusts weights
    for specified unimportant classes and re-normalizes.
    Args:
        label_source: Source of labels, can be a list of file paths or a dataset.
        num_classes (int): The total number of classes.
        ignore_index (Optional[int]): Index to ignore in the labels.
        source_type (str): Type of source, 'files' or 'dataset'.
        unimportant_class_indices (Optional[List[int]]): Indices of unimportant classes.
        target_unimportant_weight (float): Target weight for unimportant classes.
        normalize_target_sum (float): Normalize weights sum.  If -1, normalize to num_classes.
    Returns:
        torch.Tensor: A tensor of class weights.
    """

    counts = torch.zeros(num_classes, dtype=torch.float64)
    total_valid_pixels = 0

    iterator = None
    num_labels = 0
    if source_type == 'files': 
        iterator = label_source
        num_labels = len(label_source)
    elif source_type == 'dataset':
        iterator = range(len(label_source))
        num_labels = len(label_source)
    else: 
        raise ValueError("source_type must be either 'files' or 'dataset'")
    
    print(f"Processing {num_labels} labels...")
    pbar = tqdm(iterator, total=num_labels)
    for idx_or_path in pbar:
        label_data = None
        if source_type == 'files':
            path = idx_or_path; img = Image.open(path)
            label_data = torch.from_numpy(np.array(img))
        elif source_type == 'dataset':
                _, label_data = label_source[idx_or_path]; 
                label_data = torch.tensor(label_data) if not isinstance(label_data, torch.Tensor) else label_data

        label_long = label_data.long().view(-1)
        
        if ignore_index is not None: 
            valid_mask = (label_long != ignore_index)
            label_valid = label_long[valid_mask]
        else:
            label_valid = label_long
        label_valid = torch.clamp(label_valid, 0, num_classes - 1)
        
        if label_valid.numel() > 0:
                counts += torch.bincount(label_valid, minlength=num_classes).double()
                total_valid_pixels += label_valid.numel()

    print("\nFinished counting.")
    print(f"Raw pixel counts per class: {counts.long().tolist()}")
    print(f"Total valid pixels counted: {total_valid_pixels}")

    frequencies = counts / total_valid_pixels
    epsilon = 1e-6
    inverse_frequencies = 1.0 / (frequencies + epsilon)

    weights = inverse_frequencies

    if unimportant_class_indices:
        for idx in unimportant_class_indices:
            weights[idx] = min(weights)

    target_sum = normalize_target_sum if normalize_target_sum > 0 else float(num_classes)
    final_weights = weights / weights.sum() * target_sum

    print(f"Calculated Final Class Weights: {final_weights.tolist()}")

    return final_weights.float()


def convert_rgb_label_to_classes(label_array_rgb):
    """
    Converts a 3-channel RGB label map to a 1-channel class map.

    Mapping:
    [0, 0, 0] (Black)     -> 0 (Background)
    [128, 0, 0] (Red)       -> 1 (Cat)
    [0, 128, 0] (Green)     -> 2 (Dog)
    [255, 255, 255] (White) -> 0 (Background) - Assuming white is also background
    Other                 -> 255 (Ignore)

    Args:
        label_array_rgb (np.ndarray): A HxWx3 NumPy array (uint8).

    Returns:
        np.ndarray: A HxW NumPy array (uint8) with class indices.
    """
    # Input validation
    if label_array_rgb.ndim != 3 or label_array_rgb.shape[2] != 3:
        raise ValueError(
            "Input label must be 3-channel RGB (HxWx3), "
            f"but got shape {label_array_rgb.shape}"
        )

    h, w, _ = label_array_rgb.shape
    # Initialize with ignore value (255)
    label_map_1channel = np.full((h, w), 255, dtype=np.uint8)

    # Define colors as tuples for comparison
    black = (0, 0, 0)
    red = (128, 0, 0)
    green = (0, 128, 0)
    white = (255, 255, 255)

    # Create boolean masks for each color
    # Comparing tuples is faster for multi-channel exact matches typically
    mask_black = np.all(label_array_rgb == black, axis=2)
    mask_red = np.all(label_array_rgb == red, axis=2)
    mask_green = np.all(label_array_rgb == green, axis=2)
    mask_white = np.all(label_array_rgb == white, axis=2)

    # Apply mapping (order can matter if masks could overlap, but shouldn't here)
    # Map backgrounds first
    label_map_1channel[mask_black] = 0
    label_map_1channel[mask_white] = 0 # Map white to background class 0
    # Map foreground classes
    label_map_1channel[mask_red] = 1   # Cat
    label_map_1channel[mask_green] = 2 # Dog
    # Any remaining pixels stay 255

    return label_map_1channel