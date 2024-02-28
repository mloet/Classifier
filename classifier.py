# Please place any imports here.
# BEGIN IMPORTS

import numpy as np
import cv2
import random
import scipy
import scipy.ndimage

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, datasets

def model_train(net, inputs, labels, criterion, optimizer):
    
    labels = labels.reshape(-1)
    optimizer.zero_grad()
    outputs = net(inputs)
    
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss = loss.item()
    total_images = inputs.shape[0]
    maxes = torch.argmax(outputs,1)
    num_correct = (maxes == labels).int().sum()

    return running_loss, num_correct, total_images

# Data augmentation functions
class Shift(object):
    def __init__(self, max_shift=10):
        self.max_shift = max_shift

    def __call__(self, image):
        image = image.numpy()
        _, H, W = image.shape
        rand_x = random.randint(-self.max_shift, self.max_shift)
        rand_y = random.randint(-self.max_shift, self.max_shift)
        
        shift_x = np.roll(image, rand_x, axis=2)
        if rand_x >= 0:
            shift_x[:,:,:rand_x] = 0
        else:
            shift_x[:,:,rand_x:] = 0

        shift_y = np.roll(shift_x, rand_y, axis=1)
        if rand_y >= 0:
            shift_y[:,:rand_y,:] = 0
        else:
            shift_y[:,rand_y:,:] = 0

        return torch.Tensor(shift_y)

    def __repr__(self):
        return self.__class__.__name__

class Contrast(object):
    def __init__(self, min_contrast=0.3, max_contrast=1.0):
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast

    def __call__(self, image):
        image = image.numpy()
        _, H, W = image.shape

        rand_contrast = random.uniform(self.min_contrast, self.max_contrast)
        per_channel_mean = np.mean(image,axis=(1,2))
        mean_img = np.zeros_like(image)
        mean_img[0,:,:] = per_channel_mean[0]
        mean_img[1,:,:] = per_channel_mean[1]
        mean_img[2,:,:] = per_channel_mean[2]
        new_img  = mean_img + rand_contrast*(image-mean_img)
        new_clipped_img = np.clip(new_img,0,1)

        return torch.Tensor(new_clipped_img)

    def __repr__(self):
        return self.__class__.__name__

class Rotate(object):
    def __init__(self, max_angle=10):
        self.max_angle = max_angle

    def __call__(self, image):
        image = image.numpy()
        _, H, W  = image.shape

        rand_angle = random.uniform(-self.max_angle, self.max_angle)
        rotated_image = scipy.ndimage.rotate(image, rand_angle, axes=(2, 1), reshape=False, order=1)

        return torch.Tensor(rotated_image)

    def __repr__(self):
        return self.__class__.__name__

class HorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        image = image.numpy()
        _, H, W = image.shape

        rand_num = random.uniform(0,1)
        if rand_num <= self.p:
            image = np.flip(image, axis=2).copy()

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

# Model
def get_classifier_settings(net):
    dataset_means = [123./255., 116./255.,  97./255.]
    dataset_stds  = [ 54./255.,  53./255.,  52./255.]

    t_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        
        transforms.Normalize(dataset_means, dataset_stds)
    ])
    
    v_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(dataset_means, dataset_stds)
    ])
    
    batch_size = 64
    epochs = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001) # Tried a few different algorithms, landed on Adam as most accurate for the purpose
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1) # Learning rate scheduling to accelerate training & improve final accuracy

    return [t_transform, v_transform], batch_size, epochs, criterion, optimizer

class AnimalClassifierNet(nn.Module): # Keeping model size to below 5MB to analyze changes in accuracy results
    def __init__(self, num_classes=16):
        super(AnimalClassifierNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 18, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2) # Pooling to control overfitting
        self.bn1 = nn.BatchNorm2d(18) # Batch normalization to stabilize learning and accelerate model training
        self.conv2 = nn.Conv2d(18, 36, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2) 
        self.bn2 = nn.BatchNorm2d(36)
        self.conv3 = nn.Conv2d(36, 72, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(72)
        self.conv4 = nn.Conv2d(72, 144, 3, stride=1, padding=1) # Extra convolutional layers for ability to learn more complex patterns
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bn4 = nn.BatchNorm2d(144)
        self.conv5 = nn.Conv2d(144, 288, 3, stride=1, padding=1) 
        self.pool5 = nn.MaxPool2d(2, 2)
        self.bn5 = nn.BatchNorm2d(288)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(18 * 8 * 8, 576)
        self.fc2 = nn.Linear(576, num_classes) 
        self.dropout = nn.Dropout(0.5) # Dropout to combat risk of overfitting from extra layers and channels

    def forward(self, x):
        x = x.contiguous().view(-1, 3, 64, 64).float()

        x = self.relu(self.bn1(self.pool1(self.conv1(x))))
        x = self.relu(self.bn2(self.pool2(self.conv2(x))))
        x = self.relu(self.bn3(self.pool3(self.conv3(x))))
        x = self.relu(self.bn4(self.pool4(self.conv4(x))))
        x = self.relu(self.bn5(self.pool5(self.conv5(x))))
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Adversarial images
def get_adversarial(img, output, label, net, criterion, epsilon):

    loss = criterion(output, label)
    net.zero_grad()
    loss.backward()
    gradients = img.grad
    signs = torch.sign(gradients)
    noise = epsilon*signs
    perturbed_image = torch.clamp(img+noise, 0, 1)

    return perturbed_image, noise

