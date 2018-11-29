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
os.makedirs('orig_images', exist_ok=True)

channels = 1
img_size = 28
latent_dim = 100
cond_dim = 1
batch_size = 32
learning_rate = .0002
b1 = .5
b2 = .999
sample_interval = 500
orig_images_sampling_rate = 500
n_epochs = 200
img_shape = (channels, img_size, img_size)
gf_dim = 32 # img_size

print_shapes = False

cuda = True if torch.cuda.is_available() else False

def getEmbeddingFromLabel(label):
    return np.array([1]*cond_dim)

def getRandomLabel():
    return 'apple'

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        # if nc == 1:
        #     # return nn.LeakyReLU(0.2, inplace=True)
        #     return F.sigmoid(x[:])
        # else:
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])

def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)

# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block

class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 1),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img

class InspiredGen(nn.Module): # INIT_STAGE_G
    def __init__(self, gen_image, force_original_size):
        super(InspiredGen, self).__init__()
        self.gf_dim = gf_dim * 16
        self.in_dim = cond_dim+latent_dim

        self.gen_image = gen_image
        self.force_original_size = force_original_size

        self.define_module()

    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

        padding = 1
        kernel_size = -1 * ((img_size - 64) - 2 * padding) + 1
        self.reshaper = nn.Conv2d(ngf // 16, ngf // 16, kernel_size=kernel_size,
                         stride=1, padding=padding, bias=False)
        self.get_image = GET_IMAGE_G(ngf // 16)

    def forward(self, z_code, c_code):
        """
        :param z_code: batch x latent_dim
        :param c_code: batch x cond_dim
        :return: batch x ngf/16 x 64 x 64 OR image
        """
        c_z_code = torch.cat((c_code, z_code), 1)
        # state size ngf x 4 x 4
        out_code = self.fc(c_z_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size ngf/3 x 8 x 8
        out_code = self.upsample1(out_code)
        # state size ngf/4 x 16 x 16
        out_code = self.upsample2(out_code)
        # state size ngf/8 x 32 x 32
        out_code32 = self.upsample3(out_code)
        # state size ngf/16 x 64 x 64
        out_code64 = self.upsample4(out_code32)

        output = out_code64
        if self.gen_image:
            if print_shapes:
                print("out_code64=", out_code64.shape)
            if self.force_original_size:
                output = self.reshaper(output)
            output = self.get_image(output)

        return output

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [  nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(cond_dim+latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, label_embedding):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((label_embedding, noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(cond_dim + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, img, label_embedding):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((label_embedding, img.view(img.size(0), -1)), -1)
        validity = self.model(d_in)
        return validity

# Loss functions
adversarial_loss = torch.nn.MSELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = InspiredGen(True, True) # Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(b1, b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row**2, latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    label_embeddings = np.array([getEmbeddingFromLabel(getRandomLabel()) for i in range(n_row**2)])
    label_embeddings = Variable(FloatTensor(label_embeddings))
    gen_imgs = generator(z, label_embeddings)
    save_image(gen_imgs.data, 'images/%d.png' % batches_done, nrow=n_row, normalize=True)

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

for epoch in range(n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        if epoch == 0 and i % orig_images_sampling_rate == 0:
            #### save the first images
            orig_img = imgs.type(FloatTensor)
            orig_img = orig_img.view(orig_img.size(0),*img_shape)
            save_image(orig_img.data, 'orig_images/%d.png' % i, nrow=batch_size//8, normalize=True)

        batch_size = len(imgs)

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor)) # 32x794
        labels = Variable(labels.type(LongTensor))

#         # -----------------
#         #  Train Generator
#         # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        label_embeddings = np.array([getEmbeddingFromLabel(labels[i]) for i in range(batch_size)])
        label_embeddings = Variable(FloatTensor(label_embeddings))
        # Generate a batch of images
        gen_imgs = generator(z, label_embeddings)
        if print_shapes:
            print("gen_imgs=", gen_imgs.shape)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, label_embeddings)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, label_embeddings)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), label_embeddings)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, n_epochs, i, len(dataloader),
                                                            d_loss.item(), g_loss.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:
             sample_image(n_row=5, batches_done=batches_done)
