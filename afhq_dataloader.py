import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os


class afhq_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, image_size=64, transform=None):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.transform = transform

        self.img_list = os.listdir(self.dataset_path)

    def __len__(self):
        return len(os.listdir(self.dataset_path))

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dataset_path, self.img_list[idx]))
        if self.transform:
            img = self.transform(img)
        return img