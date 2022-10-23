


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""# Prepare data"""

from torch.utils.data import TensorDataset, DataLoader
import cv2
data = np.load('TibetanMNIST.npz')


image_list = []
image_int = data['image'].astype('float64')
for img in image_int:
  a = np.where(img > 128, 1, 0)
  a = a.reshape(1, a.shape[0], a.shape[1])
  image_list.append(a)
image_np = np.array(image_list)

indices = np.random.permutation(image_np.shape[0])
split_rate = 0.8
training_idx, test_idx = indices[:int(image_np.shape[0]*split_rate)], indices[int(image_np.shape[0]*split_rate):]
train_x, val_x = image_np[training_idx,:], image_np[test_idx,:]
train_y, val_y = data['label'][training_idx], data['label'][test_idx]

batch_size = 4096
train_dataset = TensorDataset(torch.from_numpy(train_x) ,torch.from_numpy(train_y)) 
val_dataset = TensorDataset(torch.from_numpy(val_x) ,torch.from_numpy(val_y)) 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

"""# Model"""

class VAE(nn.Module):
    def __init__(self, imgChannels=1, featureDim=32*20*20, zDim=256):
        super(VAE, self).__init__()

        self.encConv1 = nn.Conv2d(imgChannels, 16, 5)
        self.encConv2 = nn.Conv2d(16, 32, 5)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(32, 16, 5)
        self.decConv2 = nn.ConvTranspose2d(16, imgChannels, 5)

    def encoder(self, x):
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        x = x.view(-1, 32*20*20)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        x = F.relu(self.decFC1(z))
        x = x.view(-1, 32, 20, 20)
        x = F.relu(self.decConv1(x))
        x = torch.sigmoid(self.decConv2(x))
        return x

    def forward(self, x):

        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        #print(z.size())
        #print(z)
        out = self.decoder(z)
        return out, mu, logVar

"""# Training"""

learning_rate = 1e-3
num_epochs = 800

net = VAE().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
train_losses_his = []
valid_losses_his = []
min_loss = 100

import copy
scale = 100
for epoch in range(num_epochs):
    total_loss = 0
    total_valloss = 0
    for idx, imagedata in enumerate(train_loader, 0):
        imgs, _ = imagedata
        imgs = imgs.to(device)
        imgs = imgs*1.0
        #print(imgs.shape)
        out, mu, logVar = net(imgs)
        kl_divergence = scale * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        #print(imgs.size(), out.size())
        loss = F.binary_cross_entropy(out, imgs, size_average =False) + kl_divergence

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    
    net.eval()

    with torch.no_grad():
        for imagedata in val_loader:
            imgs, _ = imagedata
            imgs = imgs.to(device)
            imgs = imgs*1.0
            out, mu, logVAR = net(imgs)
            kl_divergence = 0.5 * torch.sum(-1 - logVAR + mu.pow(2) + logVAR.exp())
            loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence
            total_valloss+=loss.item()
        val_loss = total_valloss/(batch_size*len(val_loader))
        if val_loss<min_loss:
            min_loss = val_loss
            best_model = copy.deepcopy(net)
        if not epoch % 10  and epoch != 0:
          train_losses_his.append(total_loss/(batch_size*len(train_loader)))
          valid_losses_his.append(total_valloss/(batch_size*len(val_loader)))
    print(f'| end of epoch {epoch:3d} | training loss: {total_loss/(batch_size*len(train_loader)):5.2f} |valid loss: {val_loss:5.2f} |')

