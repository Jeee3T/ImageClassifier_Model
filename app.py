# loading the saved model and passing image


from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np

from torchvision import transforms

import PIL.Image as Image

classes = [
    "ants",
    "bees",
]

# loading model
model = torch.load('best_model.pth')

# transform the image
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

image_transforms = transforms.Compose([
    transforms.Resize((224, 224,)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))

])


def classify(model, image_transforms, image_path, classes):
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)

    output = model(image)

    #calculating probability of one img with respect to the other image class
    acc = nn.functional.softmax(output, dim=1)
    final_acc=acc*100

    print(final_acc)



    #displaying the one with maximum accuracy
    _, predicted = torch.max(output.data, 1)
    print(classes[predicted.item()])






fname = 'sampleimg/beeimage_2.jpg'

classify(model, image_transforms, fname, classes)

