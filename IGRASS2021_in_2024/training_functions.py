import torch
from tqdm import tqdm
from utils import count_trainable_parameters, plot_training_trends, save_to_json

# Models
from models.non_quantum.CNN_orig import CNN_Classifier
from models.quantum_quiskit.QCNN import QCNN_Classifier
from models.non_quantum.NN4EO import NN4EO
from models.quantum_quiskit.QNN4EO_orig import QNN4EO

def stardard_train(model_name, train_loader, val_loader, epochs, batch_size, img_shape, save_path, device):

    if model_name == 'cnn':
        model = CNN_Classifier(img_shape).to(device)
    elif model_name == 'qcnn':
        model = QCNN_Classifier(img_shape).to(device)
    elif model_name == 'nn4eo':
        model = NN4EO().to(device)
    elif model_name == 'qnn4eo':
        model = QNN4EO().to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Please choose a valid model name.")

    trainable_param = count_trainable_parameters(model)
    print(f'The model has: {trainable_param}')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    history = {
        'model_name': [model_name],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'train_param': [trainable_param],
        'ep': [epochs],
        'bs': [batch_size],
        'img_shape': [img_shape]
    }

    best_val_accuracy = 0.0 

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        train_loop = tqdm(train_loader, desc='Training', leave=False)
        
        for data, target in train_loop:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            print(output, target)
            exit(0)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * data.size(0)  # Accumulate loss
            
            train_loop.set_postfix({'Batch Loss': loss.item()})

        train_loss = running_train_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0

        val_loop = tqdm(val_loader, desc='Validating', leave=False)
        
        with torch.no_grad():
            for data, target in val_loop:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target.float())
                running_val_loss += loss.item() * data.size(0)  # Accumulate loss
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                val_loop.set_postfix({'Batch Loss': loss.item()})

        val_loss = running_val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct / total

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model
            torch.save(model.state_dict(), save_path + 'best_model.pth')
            print(f'New best validation accuracy: {val_accuracy:.2f}%')

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f'Epoch {epoch+1}/{epochs} -> '
              f'Train Loss: {train_loss:.4f}, '
              f'Validation Loss: {val_loss:.4f}, '
              f'Validation Accuracy: {val_accuracy:.2f}%')

    # Conclude, save and print
    save_to_json(history, save_path + 'training_history.json')
    plot_training_trends(history, save_path)

    return best_model
