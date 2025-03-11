#import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from PIL import Image
import os


print("IM HERE")


def resize_with_padding(image, target_size, interpolation=Image.BILINEAR):
    """
    Resize an image while keeping the aspect ratio using interpolation and zero padding.

    Args:
        image (PIL.Image): Input image.
        target_size (tuple): Desired (height, width) of the output image.
        interpolation: Interpolation method (e.g., Image.BILINEAR, Image.NEAREST, etc.)

    Returns:
        torch.Tensor: Resized and padded image tensor.
    """
    target_h, target_w = target_size
    orig_w, orig_h = image.size

    # Determine the new size while keeping aspect ratio
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    # Resize the image with interpolation
    transform_resize = transforms.Resize((new_h, new_w), interpolation=interpolation)
    image_resized = transform_resize(image)

    # Convert to tensor
    image_tensor = transforms.ToTensor()(image_resized)

    # Compute padding (left, right, top, bottom)
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    # Apply zero padding
    image_padded = F.pad(image_tensor, (left, right, top, bottom), mode="constant", value=0)

    return image_padded


import torch
import torchvision.transforms as transforms
from PIL import Image


def revert_resize_with_padding(tensor, original_size):
    """
    Reverts the resize_with_padding operation by cropping out the padding
    and resizing back to the original dimensions.

    Args:
    - tensor (torch.Tensor): The padded and resized image tensor (C, H, W).
    - original_size (tuple): The original (height, width) before resizing.

    Returns:
    - torch.Tensor: The cropped and resized image tensor.
    """

    # Get original height and width
    orig_h, orig_w = original_size
    _, h, w = tensor.shape  # Current height and width

    # Compute aspect ratio
    aspect_ratio_orig = orig_w / orig_h
    aspect_ratio_new = w / h

    # Determine cropping dimensions
    if aspect_ratio_orig > aspect_ratio_new:  # Original was wider
        new_h = h
        new_w = int(h * aspect_ratio_orig)
    else:  # Original was taller
        new_w = w
        new_h = int(w / aspect_ratio_orig)

    # Crop the center to remove padding
    top = max((h - new_h) // 2, 0)
    left = max((w - new_w) // 2, 0)
    bottom = top + new_h
    right = left + new_w
    cropped = tensor[:, top:bottom, left:right]

    # Resize back to original dimensions
    resize = transforms.Resize((orig_h, orig_w), interpolation=transforms.InterpolationMode.BILINEAR)
    reverted = resize(cropped)

    return reverted




input_folder = "Dataset_filtered/TrainVal/color"
labels_folder = "Dataset_filtered/TrainVal/label"
output_folder = "Resized_Padded"

# Read all images and extract their sizes
"""for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):  # Process only PNG images
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)"""


# Example usage
filename = "Abyssinian_1.jpg"
output_filename = "512_" + filename

os.makedirs(output_folder, exist_ok=True)
img_path = os.path.join(input_folder, filename)
img = Image.open(img_path)  # Load an image
original_size = img.size
print(f"Original size: {original_size}")  # Original size
target_size = (512, 512)  # Set target size (height, width)

output_path = os.path.join(output_folder, output_filename)
"""
resized_padded_image = resize_with_padding(img, target_size)

# Save output
vutils.save_image(resized_padded_image, output_path)
#resized_padded_image.save(output_path)

print(resized_padded_image.shape)  # Should be (C, 256, 256)
"""
# Example Usage
resized_padded_image = Image.open(output_path)
reverted_image = revert_resize_with_padding(resized_padded_image, original_size)
revert_path = os.path.join(output_folder, "reverted" + output_filename)
vutils.save_image(reverted_image, revert_path)

