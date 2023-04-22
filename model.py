import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import animation


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Defining genearator class
class Generator(nn.Module):
    def __init__(self, z_dim=100, image_size=64, num_channels=3):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.image_size = image_size
        self.num_channels = num_channels
        self.fc = nn.Linear(z_dim, 1024 * 4 * 4)
        self.conv1 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.ConvTranspose2d(128, num_channels, 4, 2, 1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 1024, 4, 4)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.tanh(self.conv4(x))
        return x

# Defining discriminator class
class Discriminator(nn.Module):
    def __init__(self, image_size=64, num_channels=3):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(num_channels, 128, 4, 2, 1)
        self.conv2 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(1024)
        self.fc = nn.Linear(1024 * 4 * 4, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = x.view(-1, 1024 * 4 * 4)
        x = torch.sigmoid(self.fc(x))
        return x


class mapping_net(nn.Module):
    def __init__(self, inp_ch, hid_ch, out_ch):
        super(mapping_net, self).__init__()
        self.mapping = nn.Sequential(nn.Linear(inp_ch, hid_ch),   
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(hid_ch, hid_ch),
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(hid_ch, hid_ch),
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(hid_ch, out_ch),
                                        nn.LeakyReLU(0.2)
                                        )
    def forward(self, z):
        z = F.normalize(z, dim=1)
        return self.mapping(z)

class AdaIN(nn.Module):

    def __init__(self, out_channels, w_dim):
        super().__init__()

        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.style_scale_transform = nn.Linear(w_dim, out_channels)
        self.style_shift_transform = nn.Linear(w_dim, out_channels)
    
    def forward(self, img, w):
        normalised_img = self.instance_norm(img)
        style_scale = self.style_scale_transform(w).to(img.device)
        style_shift = self.style_shift_transform(w).to(img.device)
        transformed_img = style_scale[:, :, None, None] * normalised_img + style_shift[:, :, None, None]
        return transformed_img

class Noise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1), requires_grad=True)
        
    def forward(self, img):
        noise = torch.randn(img.shape[0], 1, img.shape[2], img.shape[3]).to(img.device)
        return img + self.weight.to(img.device) * noise

    
# create all networks required for styleGan
class synthesis_block(nn.Module):
    def __init__(self, w_dim, inp_channel=512, out_channel=512,kernel_size=3):
        super(synthesis_block, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(inp_channel, out_channel, kernel_size,1,1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size,1,1)
        self.adain = AdaIN(out_channel, w_dim)
        self.noise = Noise(out_channel)

    def forward(self, img, w):
        img = self.upsample(img)
        img = self.adain(F.leaky_relu(self.noise(self.conv1(img)), 0.2), w)
        img = self.adain(F.leaky_relu(self.noise(self.conv2(img)), 0.2), w)
        return img


class synthesis_network(nn.Module):
    def __init__(self, w_dim, z_dim, in_ch, hid_ch, out_ch, map_ch, kernel_size=3):
        super(synthesis_network, self).__init__()
        self.starting_const = torch.randn(1, in_ch, 4, 4).to(device)
        self.synthesis_block1 = synthesis_block(w_dim, in_ch, hid_ch, kernel_size)
        self.synthesis_block2 = synthesis_block(w_dim, hid_ch, hid_ch, kernel_size)
        self.synthesis_block3 = synthesis_block(w_dim, hid_ch, hid_ch, kernel_size)
        self.synthesis_block4 = synthesis_block(w_dim, hid_ch, out_ch, kernel_size)
        self.mapping = mapping_net(z_dim, map_ch, w_dim)
        self.ini_conv = nn.Conv2d(in_ch, in_ch, kernel_size,1,1)
        self.final_conv = nn.Conv2d(out_ch, out_ch, 1)
        self.adain = AdaIN(in_ch, w_dim)
        self.noise = Noise(in_ch)

    def forward(self, z):
        w = self.mapping(z)
        img = self.adain(self.noise(self.starting_const), w)
        img = self.adain(F.leaky_relu(self.noise(self.ini_conv(img)), 0.2), z)
        img = self.synthesis_block1(img, w)
        img = self.synthesis_block2(img, w)
        img = self.synthesis_block3(img, w)
        img = self.synthesis_block4(img, w)
        img = self.final_conv(img)
        return img


