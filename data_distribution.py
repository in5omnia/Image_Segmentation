from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


# Folder containing images
input_folder = "Dataset_filtered/TrainVal/color"
trimaps_folder = "Dataset_filtered/TrainVal/label"
output_trimaps_folder = "trimaps_normalized"

normalize_trimaps = False   # Set to True to normalize trimaps

# Load the image
#img = Image.open("Abyssinian_1.png")

# Convert to NumPy array
#pixels = np.array(img)

# Print unique pixel values
#print(np.unique(pixels))


# Define input and output folders
if normalize_trimaps:
    os.makedirs(output_trimaps_folder, exist_ok=True)

    # Process all images in the folder
    for filename in os.listdir(trimaps_folder):
        if filename.endswith(".png"):  # Process only PNG images
            img_path = os.path.join(trimaps_folder, filename)
            img = Image.open(img_path)

            # Convert to NumPy array
            pixels = np.array(img).astype(np.float32)

            # Normalize (auto-level) to 0-255
            min_val, max_val = pixels.min(), pixels.max()
            if max_val > min_val:  # Avoid division by zero
                pixels = (pixels - min_val) / (max_val - min_val) * 255

            # Convert back to uint8
            normalized_img = Image.fromarray(pixels.astype(np.uint8))

            # Save output
            output_path = os.path.join(output_trimaps_folder, filename.replace(".png", ".jpg"))
            normalized_img.save(output_path)

    print("Normalization complete! Processed images are in 'trimaps_normalized/'.")


# ------ Sizing Distribution ------ #

# Lists to store width and height
widths, heights = [], []

# Read all images and extract their sizes
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):  # Process only PNG images
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)
        w, h = img.size  # PIL gives (width, height)
        widths.append(w)
        heights.append(h)

# Convert lists to NumPy arrays
widths = np.array(widths)
heights = np.array(heights)

if widths.size <= 0 or heights.size <= 0:
    print("No image sizes were extracted. Check the image files.")

# Compute statistics
median_width, median_height = np.median(widths), np.median(heights)
mean_width, mean_height = np.mean(widths), np.mean(heights)

# Print results
print(f"Median width: {median_width}, Median height: {median_height}")
print(f"Average width: {mean_width:.2f}, Average height: {mean_height:.2f}")
"""
# Plot distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Width distribution
counts, bins, _ = axes[0].hist(widths, bins=20, color="blue", alpha=0.7, edgecolor="black")
axes[0].set_title("Width Distribution")
axes[0].set_xlabel("Width (pixels)")
axes[0].set_ylabel("Frequency")
for count, bin_x in zip(counts, bins[:-1]):  # Annotate each bar
    axes[0].text(bin_x, count + 1, str(int(count)), ha='center', fontsize=10, color='black')

# Height distribution
counts, bins, _ = axes[1].hist(heights, bins=20, color="green", alpha=0.7, edgecolor="black")
axes[1].set_title("Height Distribution")
axes[1].set_xlabel("Height (pixels)")
axes[1].set_ylabel("Frequency")
for count, bin_x in zip(counts, bins[:-1]):  # Annotate each bar
    axes[1].text(bin_x, count + 1, str(int(count)), ha='center', fontsize=10, color='black')

plt.tight_layout()
plt.show()
"""
# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
bin_size = 50  # **Set bin size to 50 pixels**

# Plot Width Distribution
counts, bins, patches = axes[0].hist(widths, bins=np.arange(0, 1001, bin_size), edgecolor='black', alpha=0.7, color='blue')
axes[0].set_xlabel("Width (pixels)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Width Distribution")
axes[0].set_xlim(0, 1000)  # **Limit x-axis to 1000 pixels**


# Annotate each bar
for count, bin_x, patch in zip(counts, bins[:-1], patches):
    if count > 0:
        axes[0].text(bin_x + (bins[1] - bins[0]) / 2, count + 50, str(int(count)),
                     ha='center', fontsize=10, color='white', bbox=dict(facecolor='black', alpha=0.6))

# Plot Height Distribution
counts, bins, patches = axes[1].hist(heights, bins=np.arange(0, 1001, bin_size), edgecolor='black', alpha=0.7, color='green')
axes[1].set_xlabel("Height (pixels)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Height Distribution")
axes[1].set_xlim(0, 1000)  # **Limit x-axis to 1000 pixels**


# Annotate each bar
for count, bin_x, patch in zip(counts, bins[:-1], patches):
    if count > 0:
        axes[1].text(bin_x + (bins[1] - bins[0]) / 2, count + 50, str(int(count)),
                     ha='center', fontsize=10, color='white', bbox=dict(facecolor='black', alpha=0.6))

# Adjust layout and show the plots
plt.tight_layout()
#plt.show()

