import matplotlib.pyplot as plt
import numpy as np
import random

# Get the labels from vector one hot encoded
def reverseOneHotEncoding(one_hot_encoded, classes):
    l = np.argmax(one_hot_encoded, axis=1)
    labels = []

    for i in range(one_hot_encoded.shape[0]):
        for j in range(len(classes)):
            if l[i] == j:
                labels.append(classes[j])
    return labels

def plotDataset(images, labels, classes, columns, rows):
    fig, axes = plt.subplots(nrows = rows, ncols = columns, figsize = (columns*4,rows*4))
    for x in range(columns):
        for y in range(rows):
            index = random.randint(0, len(images)-1)

            axes[y,x].imshow(plt.imread(images[index]))
            axes[y,x].set_title('Label: ' + str(labels[index]) + 
                                '\n Class: ' + classes[np.argmax(labels[index])])
            axes[y,x].axis(False)

    plt.show()