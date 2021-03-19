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

def plotCNNhistory(history):
    x = np.arange(0, len(history.history['accuracy']))
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
    
    axes[0].plot(x, history.history['accuracy'], '-*')
    axes[0].plot(x, history.history['val_accuracy'], '-o')
    axes[0].legend(['Training Accuracy', 'Validation Accuracy'], fontsize = 14)
    axes[0].set_title('Training and Validation Accuracy', fontsize=20)
    axes[0].set_xlabel('Epochs', fontsize = 14)
    axes[0].set_ylabel('Accuracy', fontsize = 14)
    axes[0].set_ylim([0,1])
    axes[0].grid()

    axes[1].plot(x, history.history['loss'], '-*')
    axes[1].plot(x, history.history['val_loss'], '-o')
    axes[1].legend(['Training Loss', 'Validation Loss'], fontsize = 14)
    axes[1].set_title('Training and Validation loss', fontsize = 20)
    axes[1].set_xlabel('Epochs', fontsize = 14)
    axes[1].set_ylabel('Loss', fontsize = 14)
    axes[1].set_ylim([0,None])
    axes[1].grid()

    fig.tight_layout()
    plt.show()