from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from torch.autograd import Variable

# Hyperparameters
batch_size = 64
image_size = 64
codings_size = 100

# Creating the transformations
transform = transforms.Compose([transforms.Scale(image_size), transforms.ToTensor(
), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

# Loading the dataset
dataset = dset.CIFAR10(root='./data', download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Initialising the weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(codings_size, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output


# Creating the Generator
netG = Generator()
netG.apply(weights_init)

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)


# Creating the Discriminator
netD = Discriminator()
netD.apply(weights_init)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training the Deep Convolution GAN
epochs = 25
for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):

        # Step 1a: Training the Discriminator with real images
        netD.zero_grad()
        real, _ = data
        input = Variable(real)
        target = Variable(torch.ones(input.size()[0]))
        output = netD(input)
        errD_real = criterion(output, target)

        # Step 1b: Training the Discriminator with fake images from the Generator
        noise = Variable(torch.randn(input.size()[0], codings_size, 1, 1))
        fake = netG(noise)
        target = Variable(torch.zeros(input.size()[0]))
        output = netD(fake.detach())
        errD_fake = criterion(output, target)

        # Step 1c: Backpropagate the errors
        errD = errD_fake + errD_real
        errD.backward()
        optimizerD.step()

        # Step 2a: Training the Generator
        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        output = netD(fake)
        errG = criterion(output, target)

        # Step 2b: Backpropagate the error
        errG.backward()
        optimizerG.step()

        # Step 3: Print the losses
        print("[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f" %
             (epoch, epochs, i, len(dataloader), errD.data[0], errG.data[0]))
        if (i % 100 == 0):
            vutils.save_image(real, "%s/real_samples.png" % ("./results"), normalize=True)
            fake = netG(noise)
            vutils.save_image(fake.data, "%s/fake_samples_epoch_%03d.png" % ("./results", epoch), normalize=True)