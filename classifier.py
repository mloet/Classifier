import numpy as np
import cv2
import random
import scipy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

#########################################################
# BASELINE MODEL                                        #
#########################################################


class AnimalBaselineNet(nn.Module):
    def __init__(self, num_classes=16):
        super(AnimalBaselineNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 12, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(12, 24, 3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8*8*24, 128)
        self.cls = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.contiguous().view(-1, 3, 64, 64).float()

        conv1_result = self.conv1(x)
        relu1_result = self.relu(conv1_result)
        conv2_result = self.conv2(relu1_result)
        relu2_result = self.relu(conv2_result)
        conv3_result = self.conv3(relu2_result)
        relu3_result = self.relu(conv3_result)
        fc_result = self.fc(torch.reshape(relu3_result, (x.shape[0], -1)))
        relu4_result = self.relu(fc_result)
        x = self.cls(relu4_result)
        return x


def model_train(net, inputs, labels, criterion, optimizer):
    # TODO: Foward pass
    # TODO-BLOCK-BEGIN
    labels = labels.reshape(-1)
    optimizer.zero_grad()
    outputs = net(inputs)
    # TODO-BLOCK-END

    # TODO: Backward pass
    # TODO-BLOCK-BEGIN
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss = loss.item()
    total_images = inputs.shape[0]
    maxes = torch.argmax(outputs, 1)
    num_correct = (maxes == labels).int().sum()
    # TODO-BLOCK-END

    return running_loss, num_correct, total_images

#########################################################
# DATA AUGMENTATION
#########################################################


class Shift(object):
    def __init__(self, max_shift=10):
        self.max_shift = max_shift

    def __call__(self, image):
        image = image.numpy()
        _, H, W = image.shape
        rand_x = random.randint(-self.max_shift, self.max_shift)
        rand_y = random.randint(-self.max_shift, self.max_shift)

        # shift by x
        shift_x = np.roll(image, rand_x, axis=2)
        if rand_x >= 0:
            shift_x[:, :, :rand_x] = 0
        else:
            shift_x[:, :, rand_x:] = 0

        # shift by y
        shift_y = np.roll(shift_x, rand_y, axis=1)
        if rand_y >= 0:
            shift_y[:, :rand_y, :] = 0
        else:
            shift_y[:, rand_y:, :] = 0

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
        per_channel_mean = np.mean(image, axis=(1, 2))
        mean_img = np.zeros_like(image)
        mean_img[0, :, :] = per_channel_mean[0]
        mean_img[1, :, :] = per_channel_mean[1]
        mean_img[2, :, :] = per_channel_mean[2]
        new_img = mean_img + rand_contrast*(image-mean_img)
        new_clipped_img = np.clip(new_img, 0, 1)

        return torch.Tensor(new_clipped_img)

    def __repr__(self):
        return self.__class__.__name__


class Rotate(object):
    def __init__(self, max_angle=10):
        self.max_angle = max_angle

    def __call__(self, image):
        image = image.numpy()
        _, H, W = image.shape

        rand_angle = random.uniform(-self.max_angle, self.max_angle)
        rotated_image = scipy.ndimage.rotate(
            image, rand_angle, axes=(2, 1), reshape=False, order=1)

        return torch.Tensor(rotated_image)

    def __repr__(self):
        return self.__class__.__name__


class HorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        image = image.numpy()
        _, H, W = image.shape

        rand_num = random.uniform(0, 1)
        if rand_num <= self.p:
            image = np.flip(image, axis=2).copy()

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

#########################################################
# ADVERSARIAL IMAGES
#########################################################


def get_adversarial(img, output, label, net, criterion, epsilon):
    loss = criterion(output, label)
    net.zero_grad()
    loss.backward()
    gradients = img.grad
    signs = torch.sign(gradients)
    noise = epsilon*signs
    perturbed_image = torch.clamp(img+noise, 0, 1)

    return perturbed_image, noise
