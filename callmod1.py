#to be worked on

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import PIL.Image as Image

classes=[
    "ants",
    "bees"
]


#loading model
model=torch.load('model.pth')


#transform the image
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

image_transforms= transforms.Compose([
    transforms.Resize((224,224,)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))

])

def classify(model,image_transforms,image_path,classes):
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image=  image.unsqueeze(0)

    output=model(image)
    _,predicted=torch.max(output.data,1)

    print(predicted.item())





#
# imsize = 256
# loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
#
# def image_loader(image_name):
#     """load image, returns cuda tensor"""
#     image = Image.open(image_name)
#     image = loader(image).float()
#     image = Variable(image, requires_grad=True)
#     image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
#     return image.cuda()  #assumes that you're using GPU
#
# image = image_loader('./antimage.jpg')
#
# model(image)