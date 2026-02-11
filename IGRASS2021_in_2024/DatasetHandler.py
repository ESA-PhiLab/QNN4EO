import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import glob
import os
from torch.utils.data import Dataset, DataLoader

class DatasetHandler:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.classes = glob.glob(os.path.join(dataset_path, '*'))
    
    def print_classes(self):
        print('Classes: ') 
        for i,c in enumerate(self.classes): 
            print('     Class ' + str(i) + ' ->', c)

    def load_paths_labels(self, classes):
        # Initialize imaages path and images label lists
        imgs_path = []
        imgs_label = []
        class_counter = 0
        encoded_class = np.zeros((len(classes)))

        # For each class in the class list
        for c in classes:
            # List all the images in that class
            paths_in_c = glob.glob(c+'/*')
            # For each image in that class
            for path in paths_in_c:
                # Append the path of the image in the images path list
                imgs_path.append(path)
                # One hot encode the label
                encoded_class[class_counter] = 1
                # Append the label in the iamges label list
                imgs_label.append(encoded_class)
                # Reset the class
                encoded_class = np.zeros((len(classes)))

            # Jump to the next class after iterating all the paths
            class_counter = class_counter + 1

        # Shuffler paths and labels in the same way
        c = list(zip(imgs_path, imgs_label))
        random.shuffle(c)
        imgs_path, imgs_label = zip(*c)

        return np.array(imgs_path), np.array(imgs_label)
    
    # Split the dataset into training and validation dataset
    def train_validation_split(self, images, labels, split_factor = 0.2):
        val_size = int(len(images)*split_factor)
        train_size = int(len(images) - val_size)
        return images[0:train_size], labels[0:train_size, ...], images[train_size:train_size+val_size], labels[train_size:train_size+val_size, ...]
    
  
class CustomImageDataset(Dataset):
    def __init__(self, imgs_path, imgs_label, img_shape=(64, 64, 3)):
        self.imgs_path = imgs_path
        self.imgs_label = imgs_label
        self.img_shape = img_shape

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        # Load the image and label
        img = plt.imread(self.imgs_path[idx]) / 255.0
        img = np.transpose(img, (2, 0, 1))  # Convert to (C, H, W) format
        label = np.argmax(self.imgs_label[idx])
        
        # Convert to tensors
        img_tensor = torch.Tensor(img)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return img_tensor, label_tensor

def data_loader(imgs_path, imgs_label, batch_size=1, img_shape=(64, 64, 3), shuffle=True, num_workers=4):
    dataset = CustomImageDataset(imgs_path, imgs_label, img_shape)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)
    
    return data_loader
