import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from afhq_dataloader import *
from model import *
from trainer import *

dataset_path = "/home/sachi/Desktop/iisc/data/afhq/train/dog/"

def train_dcgan(dataset_path, z_dim=100, image_size=64, num_channels=3, lr=0.001, batch_size=64, num_epochs=100,optim="Adam"):

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = afhq_dataset(dataset_path,64, transform = transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    trainer = Trainer(train_loader, z_dim, image_size, num_channels)
    generator, discriminator = trainer.train(lr, num_epochs, batch_size, optim)
    torch.save(generator.state_dict(), "dcgan_generator.pth")


def train_stylegan(dataset_path, z_dim=100, w_dim=100, image_size=64, num_channels=3, lr=0.001, batch_size=64, num_epochs=100):

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = afhq_dataset(dataset_path,64, transform = transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    trainer = stylegan_trainer(train_loader, z_dim, w_dim, in_ch=512, hid_ch=64, out_ch=3, map_ch=512)
    trainer.train(lr, num_epochs)


if __name__ == "__main__":
    train_dcgan(dataset_path,)

    train_stylegan(dataset_path)




