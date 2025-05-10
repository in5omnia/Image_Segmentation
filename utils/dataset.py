import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.io import decode_image

class dataset(Dataset):
    """
    Initializes the dataset class.
    Args:
        img_dir (str): Directory containing the images.
        label_dir (str): Directory containing the labels.
        transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
        target_transform (callable, optional): Optional transform to be applied on a target. Defaults to None.
    """
    def __init__(self, img_dir, label_dir, transform=None, target_transform=None):
        # Initialize the image and label directories
        self.img_dir = img_dir
        self.label_dir = label_dir
        # Get the names of the images, without the extension, and sort them
        self.img_names = sorted([os.path.splitext(filename)[0] for filename in os.listdir(img_dir)])
        # Calculate the length of the dataset
        self.len = len(self.img_names)
        # Initialize the transforms
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        Loads and returns an image label pair from the dataset.
        Args:
            idx (int): Index of the item to retrieve.
        Returns:
            tuple: A tuple containing the image and its corresponding label.
        """
        # Load the image and normalize it
        img = decode_image(os.path.join(self.img_dir, self.img_names[idx] + ".jpg")).float() / 255
        # Load the label
        label = decode_image(os.path.join(self.label_dir, self.img_names[idx] + ".png"))

        # Apply transformations to the image if specified
        if self.transform:
            img = self.transform(img)

        # Apply transformations to the label if specified
        if self.target_transform:
            label = self.target_transform(label)

        return img, label
    
class promptDataset(Dataset):
    """
    Initializes the promptDataset class.
    Args:
        img_dir (str): Directory containing the images.
        heatmap_dir (str): Directory containing the heatmaps.
        label_dir (str): Directory containing the labels.
        transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
        target_transform (callable, optional): Optional transform to be applied on a target. Defaults to None.
    """
    def __init__(self, img_dir, heatmap_dir, label_dir, transform=None, target_transform=None):
        # Initialize the image, heatmap, and label directories
        self.img_dir = img_dir
        self.heatmap_dir = heatmap_dir
        self.label_dir = label_dir
        # Get the names of the images, without the extension, and sort them
        self.img_names = sorted([os.path.splitext(filename)[0] for filename in os.listdir(img_dir)])
        # Calculate the length of the dataset
        self.len = len(self.img_names)
        # Initialize the transforms
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        """
        Loads and returns an image, heatmap, label triplet from the dataset.
        Args:
            idx (int): Index of the item to retrieve.
        Returns:
            tuple: A tuple containing the image, heatmap, and its corresponding label.
        """
        # Load the image and normalize it
        img = decode_image(os.path.join(self.img_dir, self.img_names[idx] + ".jpg")).float()/255
        # Load the heatmap and normalize it
        heatmap = decode_image(os.path.join(self.heatmap_dir, self.img_names[idx] + ".png"))/255
        # Load the label
        label = decode_image(os.path.join(self.label_dir, self.img_names[idx] + ".png"))

        # Apply transformations to the image if specified
        if self.transform:
            img = self.transform(img)

        # Apply transformations to the label if specified
        if self.target_transform:
            label = self.target_transform(label)
            

        return img, heatmap, label


def display_img_label(data, idx):
    """
    Displays an image and its corresponding label using matplotlib.
    Args:
        data (torch.utils.data.Dataset): The dataset containing images and labels.
        idx (int): The index of the image and label to display.
    """
    # Get the image and label from the dataset
    img, label = data[idx]
    # Create a figure with two subplots
    figure = plt.figure(figsize=(10, 20))
    # Add the first subplot for the image
    figure.add_subplot(1, 2, 1)
    # Display the image, permuting the dimensions to match matplotlib's expected format
    plt.imshow(img.permute(1, 2, 0))

    # Add the second subplot for the label
    figure.add_subplot(1, 2, 2)
    # Display the label as a grayscale image, permuting the dimensions
    plt.imshow(label.permute(1, 2, 0), cmap='grey')

    # Show the plot
    plt.show()


class target_remap(object):
    """
    Remaps boundary class (255) to 3
    """
    def __call__(self, img):
        # Remap pixel value 255 to 3
        img[img == 255] = 3
        return img


def diff_size_collate(batch):
    """
    Collates a batch of data with potentially different image sizes.
    Args:
        batch (list): A list of tuples, where each tuple contains an image and its label.
    Returns:
        tuple: A tuple containing two lists: a list of images and a list of labels.
    """
    # Extract images from the batch
    imgs = [item[0] for item in batch]
    # Extract labels from the batch
    labels = [item[1] for item in batch]
    return imgs, labels