# CNN Test
'''
Formula to calculate the output size = (W-F +2P)/S + 1
W = input width
F = filter size
P = padding
S = stride
'''

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.proj3d import transform

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epoch = 0
batch_size = 4
learning_rate = 0.001

# Dataset had PILImages images of range [0, 1]
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with mean and std
])


# Import CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create DataLoader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  # Root is where it has to be stored
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get some random training images
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Show images
imshow(torchvision.utils.make_grid(images))

conv1 = nn.Conv2d(3, 6, 5)
pool = nn.MaxPool2d(2, 2)
conv2 = nn.Conv2d(6, 16, 5)
print("Shape of the images: ")  # [4, 3, 32, 3] batch-size = 4, channels = 3 img_size = 32x32
print(images.shape)
print()

# First conv layer
x = conv1(images)
print("First conv layer")
print(x.shape)  # [4, 6, 28, 28] output_channels = 6, img_size = 28x28
print() # [(32 - 5 + 2(0)) / 1] + 1)

print("First pooling layer")
x = pool(x)
print(x.shape)  # [4, 6, 14, 14] kernel_size = 2, 2 & stride = 2 will reduce the images by factor of 2
print()

print("Second conv layer")
x = conv2(x)
print(x.shape) # [4, 16, 10, 10]
print()

print("Second pooling layer")
x = pool(x)
print(x.shape)  # [4, 16, 5, 5]  Thus, size reduces by factor of 2 after applying pooling layer?