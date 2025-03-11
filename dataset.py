import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, PILToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import os
from torchvision.io import decode_image

target_batch_size = 64
batch_size = 2


class dataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_names = sorted([os.path.splitext(filename)[0] for filename in os.listdir(img_dir)])
        self.len = len(self.img_names)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img = decode_image(os.path.join(self.img_dir, self.img_names[idx] + ".jpg")).float() / 255
        label = decode_image(os.path.join(self.label_dir, self.img_names[idx] + ".png"))

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        return img, label


def display_img_label(data, idx):
    img, label = data[idx]
    figure = plt.figure(figsize=(10, 20))
    figure.add_subplot(1, 2, 1)
    plt.imshow(img.permute(1, 2, 0))

    figure.add_subplot(1, 2, 2)
    plt.imshow(label.permute(1, 2, 0), cmap='grey')

    plt.show()


class target_remap(object):
    def __call__(self, img):
        img[img == 255] = 3
        return img


def diff_size_collate(batch):
    imgs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return imgs, labels

