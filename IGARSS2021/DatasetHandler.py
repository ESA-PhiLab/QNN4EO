import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import glob
import os

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
    
    # Data genertor: given images paths and images labels yield a batch of images and labels
    def cnn_data_loader(self, imgs_path, imgs_label, batch_size = 16, img_shape = (64, 64, 3), n_classes = 2):
        # Initialize the vectors to be yield
        batch_in = np.zeros((batch_size, img_shape[0], img_shape[1], img_shape[2]))
        batch_out = np.zeros((batch_size, n_classes))

        # Repeat until the generator will be stopped
        while True:
            # Load a batch of images and labels
            for i in range(batch_size):
                # Select a random image and labels from the dataset
                index = random.randint(0, len(imgs_path)-1)
                # Fill the vectors with images and labels
                batch_in[i, ...] = plt.imread(imgs_path[index])/255.0
                batch_out[i, ...] = imgs_label[index]

            # Yield/Return the image and labeld vectors
            yield batch_in, batch_out
    
    # Data genertor: given images paths and images labels yield a batch of images and labels
    def qcnn_data_loader(self, imgs_path, imgs_label, batch_size = 1, img_shape = (64, 64, 3)):
        # Initialize the vectors to be yield
        batch_in = np.zeros((batch_size, img_shape[2], img_shape[0], img_shape[1]))
        batch_out = np.zeros((batch_size))

        # Repeat until the generator will be stopped
        while True:
            # Load a batch of images and labels
            for i in range(batch_size):
                # Select a random image and labels from the dataset
                index = random.randint(0, len(imgs_path)-1)
                # Fill the vectors with images and labels
                batch_in[i, ...] = np.transpose(plt.imread(imgs_path[index])/255.0)
                batch_out[i] = np.argmax(imgs_label[index])
            # Yield/Return the image and labeld vectors
            yield  torch.Tensor(batch_in),  torch.Tensor(batch_out).type(torch.LongTensor)