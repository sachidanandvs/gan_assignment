import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from model import *

class Trainer(nn.Module):
    def __init__(self, train_dataset,z_dim=100, image_size=64, num_channels=3):
        super(Trainer, self).__init__()
        # initialize important parameters and model, optimizer
        self.z_dim = z_dim
        self.image_size = image_size
        self.num_channels = num_channels
        self.optim = optim
        self.train_dataset = train_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator(z_dim, image_size, num_channels).to(self.device)
        self.discriminator = Discriminator(image_size, num_channels).to(self.device)
        

    # Code for training the model
    def train(self,lr=0.001,num_of_epoch=20,batch_size=64, optim="Adam"):
        generated_samples = []
        discriminator_loss = []
        generator_loss = []
        discriminator_accuracy = []
        criterion = nn.BCELoss()
        if(optim == "Adam"):
            d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
            g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr)
        elif(optim == "SGD"):
            d_optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=lr)
            g_optimizer = torch.optim.SGD(self.generator.parameters(), lr=lr)
        dlr_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=20, gamma=0.3)
        glr_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=20, gamma=0.3)

        # ToDO add one more optimisers

        num_of_iter = self.train_dataset.__len__() // next(iter(self.train_dataset)).shape[0]
        print("Number of iterations: ", num_of_iter)
        for epoch in range(num_of_epoch):
            for i, images in enumerate(self.train_dataset):
                batch_size = images.size(0)
  
                images = images.to(self.device)
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                outputs = self.discriminator(images)
                d_loss_real = criterion(outputs, real_labels)
                real_score = outputs

                z = torch.randn(batch_size, self.z_dim).to(self.device)
                fake_images = self.generator(z)
                
                outputs = self.discriminator(fake_images)
                d_loss_fake = criterion(outputs, fake_labels)
                fake_score = outputs
                d_loss = d_loss_real + d_loss_fake

                discriminator_loss.append(d_loss.item())
            
                self.discriminator.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                z = torch.randn(batch_size, self.z_dim).to(self.device)
                fake_images = self.generator(z).to(self.device)
                outputs = self.discriminator(fake_images)
                g_loss = criterion(outputs, real_labels)

                generator_loss.append(g_loss.item())
                self.generator.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                discriminator_accuracy.append((real_score.mean().item() + fake_score.mean().item()) / 2)
                
                if (i + 1) % 5 == 0:
                    print('Epoch [%d/%d], Step [%d/%d], d_loss: %.4f, g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f'
                          % (epoch, num_of_epoch, i + 1, num_of_iter, d_loss.item(), g_loss.item(),
                             real_score.mean().item(), fake_score.mean().item()))
            dlr_scheduler.step()
            glr_scheduler.step()
            # downsample image for better visualization
            generated_samples.append(fake_images[0].detach().cpu().numpy())
        
        # Plotting the loss and accuracy
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(generator_loss, label="G")
        plt.plot(discriminator_loss, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("loss_{0}_{1}_{2}.png".format(optim,lr,num_of_epoch))

        plt.figure(figsize=(10, 5))
        plt.title("Discriminator Accuracy During Training")
        plt.plot(discriminator_accuracy, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("accuracy_{0}_{1}_{2}.png".format(optim,lr,num_of_epoch))

        # Plotting the generated images
        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in generated_samples]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        ani.save("gan_{0}_{1}_{2}.gif".format(self.optim,lr,num_of_epoch))
        


        return self.generator, self.discriminator


class stylegan_trainer(nn.Module):
    def __init__(self, train_dataset, z_dim, w_dim, in_ch,hid_ch,out_ch, map_ch):
        super(stylegan_trainer, self).__init__()
        self.z_dim = z_dim
        self.train_dataset = train_dataset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator = synthesis_network(w_dim,z_dim,in_ch,hid_ch,out_ch,map_ch).to(self.device)
        self.discriminator = Discriminator(64, 3).to(self.device)

    def train(self,lr, epoches):
        generated_samples = []
        discriminator_loss = []
        generator_loss = []
        discriminator_accuracy = []
        criterion = nn.BCELoss()
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        g_param = []
        for param in self.generator.parameters():
            if(param.requires_grad):
                g_param.append(param)
        g_optimizer = torch.optim.Adam(g_param, lr=lr)
        dlr_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=20, gamma=0.3)
        glr_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=20, gamma=0.3)

        num_of_iter = len(self.train_dataset)
        print("Number of iterations: ", num_of_iter)

        for epoch in range(epoches):
            for i, images in enumerate(self.train_dataset):
                batch_size = images.size(0)

                images = images.to(self.device)
                real_labels = torch.ones(batch_size,1).to(self.device)
                fake_labels = torch.zeros(batch_size,1).to(self.device)
                outputs = self.discriminator(images)
                d_loss_real = criterion(outputs, real_labels)
                real_score = outputs

                z = torch.randn(batch_size, self.z_dim).to(self.device)
                fake_images = self.generator(z)
                outputs = self.discriminator(fake_images)
                d_loss_fake = criterion(outputs, fake_labels)
                fake_score = outputs
                d_loss = d_loss_real + d_loss_fake

                discriminator_loss.append(d_loss.item())
            
                self.discriminator.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                z = torch.randn(batch_size, self.z_dim).to(self.device)
                fake_images = self.generator(z)
                outputs = self.discriminator(fake_images)
                g_loss = criterion(outputs, real_labels)

                generator_loss.append(g_loss.item())
                self.generator.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                discriminator_accuracy.append((real_score.mean().item() + fake_score.mean().item()) / 2)
                
                if (i + 1) % 5 == 0:
                    print('Epoch [%d/%d], Step [%d/%d], d_loss: %.4f, g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f'
                          % (epoch, epoches, i + 1, num_of_iter, d_loss.item(), g_loss.item(),
                             real_score.mean().item(), fake_score.mean().item()))
            
            dlr_scheduler.step()
            glr_scheduler.step()
            # downsample image for better visualization
            generated_samples.append(fake_images[0].detach().cpu().numpy())

        # save the model checkpoints
        torch.save(self.generator.state_dict(), 'stylegan_generator.pth')

        # plot the loss and accuracy
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(generator_loss,label="G")
        plt.plot(discriminator_loss,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.savefig('style_gan_loss.png')

        plt.figure(figsize=(10,5))
        plt.title("Discriminator Accuracy During Training")
        plt.plot(discriminator_accuracy)
        plt.xlabel("iterations")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
        plt.savefig('style_gan_accuracy.png')

        # visualize the generated images
        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in generated_samples]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        ani.save("style_gan.gif", writer='imagemagick', fps=1)






