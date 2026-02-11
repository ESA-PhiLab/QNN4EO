import torch, time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from utils import save_to_json


def evaluate_model(model, val_loader, device, save_path):
    
    results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'avg_inference_time': [] 
    }

    predictions = []
    ground_truth = []
    inference_time = []

    model.eval()
    
    evaluation_loop = tqdm(val_loader, desc='Evaluation', leave=False)
    
    with torch.no_grad():
        for data, target in evaluation_loop:

            data, target = data.to(device), target.to(device)

            # Measure the inference time
            start_time = time.time()
            output = model(data)
            end_time = time.time()
            
            # pred = output.argmax(dim=1, keepdim=True) # Quiskit
            pred = (output >= 0.5).long()  # HQM


            predictions.append(pred.item())
            ground_truth.append(target.item())
            inference_time.append(end_time - start_time) 

        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)

        cm = confusion_matrix(predictions, ground_truth)

        accuracy = (cm[0,0] + cm[1,1]) / sum(sum(cm)) * 100
        precision = (cm[0,0] + cm[1,1])/((cm[0,0] + cm[1,1])+cm[0,1]) * 100
        recall = (cm[0,0] + cm[1,1])/((cm[0,0] + cm[1,1])+cm[1,0]) * 100
        f1 = 2 * (precision * recall)/(precision+recall)

    results['accuracy'].append(accuracy)
    results['precision'].append(precision)
    results['recall'].append(recall)
    results['f1'].append(f1) 

    print('Accuracy: %.2f %%' % accuracy)
    print('Precision: %.2f %%' % precision)
    print('Recall: %.2f %%' % recall)
    print('F1 score: %.2f %%' % f1)

    # Calculate and print average inference time
    avg_inference_time = np.mean(inference_time)
    results['avg_inference_time'].append(avg_inference_time)

    # Conclude, save and print
    save_to_json(results, save_path + 'results.json')

    return results