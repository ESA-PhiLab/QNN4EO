import torch, random, json
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_training_trends(history, save_path):
    
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot training & validation loss values
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue', marker='o')
    plt.plot(epochs, history['val_loss'], label='Validation Loss', color='red', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save the loss plot as a PDF file
    plt.savefig(save_path + 'loss_trends.pdf', format='pdf')
    plt.show()

    # Plot validation accuracy values
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy', color='green', marker='o')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the accuracy plot as a PDF file
    plt.savefig(save_path + 'accuracy_trend.pdf', format='pdf')
    plt.show()


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_to_json(info, save_path):
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=4)

