import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs('images', exist_ok=True)

channels = 1
img_size = 28
n_classes = 1
latent_dim = 100
batch_size = 32
lr = .0002
b1 = .5
b2 = .999
n_epochs = 200
print_shapes = False
sample_interval = 512

img_shape = (channels, img_size, img_size)

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, latent_dim)

        self.init_size = img_size // 4 # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128*self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        if print_shapes:
            print("-----start gen forward")
        gen_input = torch.mul(self.label_emb(labels), noise)
        if print_shapes:
            print("gen_input=", gen_input.shape)
        out = self.l1(gen_input)
        if print_shapes:
            print("out1=", out.shape)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        if print_shapes:
            print("out2=", out.shape)
        img = self.conv_blocks(out)
        if print_shapes:
            print("img=", img.shape)
            print("------end gen forward")
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        #ds_size = img_size // 2**4
        ds_size = img_size
        for i in range(4):
            ds_size = (ds_size - 1) // 2 + 1
        # Output layers
        self.adv_layer = nn.Sequential( nn.Linear(128*ds_size**2, 1),
                                        nn.Sigmoid())
        self.aux_layer = nn.Sequential( nn.Linear(128*ds_size**2, n_classes),
                                        nn.Softmax())

    def forward(self, img):
        if print_shapes:
            print("++++start desc forward")
            print("img=", img.shape)
        out = self.conv_blocks(img)
        if print_shapes:
            print("out1=", out.shape)
        out = out.view(out.shape[0], -1)
        if print_shapes:
            print("out2=", out.shape)
        validity = self.adv_layer(out)
        if print_shapes:
            print("validity=", validity.shape)
        label = self.aux_layer(out)
        if print_shapes:
            print("label=", label.shape)
            print("++++end desc forward")

        return validity, label

# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
#os.makedirs('../../data/mnist', exist_ok=True)
#dataloader = torch.utils.data.DataLoader(
#    datasets.MNIST('../../data/mnist', train=True, download=True,
#                   transform=transforms.Compose([
#                        transforms.Resize(img_size),
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                   ])),
#    batch_size=batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
    labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, 'images/%d.png' % batches_done, nrow=n_row, normalize=True)

# read the data from data folder

class QuickDrawDataset(Dataset):
    """Quick Draw dataset."""

    def __init__(self, label, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = np.load(f'data/{label}.npy')
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        image = self.data_frame[idx]
        label = self.label
        return image, 1

data = QuickDrawDataset(label='apple', transform=transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)



# ----------
#  Training
# ----------

for epoch in range(n_epochs):
    print("epoch=", epoch)
    for i, (imgs, labels) in enumerate(dataloader):
        if print_shapes:
            print("batch=", i)
        batch_size = imgs.shape[0]
        if print_shapes:
            print("imgs=", imgs.shape)

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        if print_shapes:
            print("valid=", valid.shape)
            print("fake=", fake.shape)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        real_imgs = real_imgs.view(batch_size,channels,img_size,img_size)
        labels = Variable(labels.type(LongTensor))
        if print_shapes:
            print("real_imps=", real_imgs.shape)
            print("labels=", labels.shape)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))
        if print_shapes:
            print("z=", z.shape)
            print("gen_labels=", gen_labels.shape)

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)
        if print_shapes:
            print("gen_imgs=", gen_imgs.shape)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)
        if print_shapes:
            print("validity=", validity.shape)
            print("pred_label=", pred_label.shape)
            print("g_loss=", g_loss.shape)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        if print_shapes:
            print("real_pred=", real_pred)
            print("valid=", valid)
            print("real_aux=", real_aux)
            print("labels=", labels)
        d_real_loss = adversarial_loss(real_pred, valid)
        if print_shapes:
            print("d_real_loss=", d_real_loss.shape)

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss =  adversarial_loss(fake_pred, fake)
        if print_shapes:
            print("fake_pred=", fake_pred.shape)
            print("fake_aux=", fake_aux.shape)
            print("d_fake_loss=", d_fake_loss.shape)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]" % (epoch, n_epochs, i, len(dataloader),
                                                            d_loss.item(), 100 * d_acc,
                                                            g_loss.item()))
        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
