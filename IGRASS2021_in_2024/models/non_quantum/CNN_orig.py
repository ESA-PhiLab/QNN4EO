import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Classifier(nn.Module):
    def __init__(self, img_shape, n_classes=2):
        super(CNN_Classifier, self).__init__()
        self.img_shape = img_shape
        self.n_classes = n_classes

        kernel_size = 5
        stride_size = 1
        pool_size = 2
        padding_size = 2  # Padding to maintain the size after convolution

        self.conv1 = nn.Conv2d(in_channels=self.img_shape[0], out_channels=3, kernel_size=kernel_size, stride=stride_size, padding=padding_size)
        self.pool1 = nn.MaxPool2d(kernel_size=pool_size, stride=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=kernel_size, stride=stride_size, padding=padding_size)
        self.pool2 = nn.MaxPool2d(kernel_size=pool_size, stride=2)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=kernel_size, stride=stride_size, padding=padding_size)
        self.pool3 = nn.MaxPool2d(kernel_size=pool_size, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=self.calculate_flattened_size(), out_features=64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(in_features=64, out_features=self.n_classes)

    def calculate_flattened_size(self):
        with torch.no_grad():
            x = torch.randn(1, *self.img_shape) 
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            x = self.flatten(x)
            return x.numel()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

