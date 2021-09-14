from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
from torch.autograd import Function
from qiskit.visualization import *
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import qiskit
import torch

class QuantumCircuit:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """

    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        
        all_qubits = [i for i in range(n_qubits)]
        self.theta = qiskit.circuit.Parameter('theta')
        
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        
        self._circuit.measure_all()
        # ---------------------------

        self.backend = backend
        self.shots = shots
    
    def run(self, thetas):
        job = qiskit.execute(self._circuit, 
                            self.backend, 
                            shots = self.shots,
                            parameter_binds = [{self.theta: theta} for theta in thetas])
        result = job.result().get_counts(self._circuit)
        
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        
        # Compute probabilities for each state
        probabilities = counts / self.shots
        # Get state expectation
        expectation = np.sum(states * probabilities)
        
        return np.array([expectation])

class HybridFunction(Function):
    """ Hybrid quantum - classical function definition """
    
    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z = ctx.quantum_circuit.run(input[0].tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(input, result)

        return result
        
    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())
        
        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift
        
        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left  = ctx.quantum_circuit.run(shift_left[i])
                
            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            gradients.append(gradient)
            gradients = np.array([gradients]).T
        return torch.tensor([gradients]).float() * grad_output.float(), None, None

class Hybrid(nn.Module):
    """ Hybrid quantum - classical layer definition """
    
    def __init__(self, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(1, backend, shots)
        self.shift = shift
        
    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)

class QNN4EO(nn.Module):
    def __init__(self):
        super(QNN4EO, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(2704 , 64)#256
        self.fc2 = nn.Linear(64, 1)
        self.hybrid = Hybrid(qiskit.Aer.get_backend('qasm_simulator'), 100, np.pi / 2)

        self.loss_function = nn.NLLLoss()
        self.optimizer = optim.Adam(self.parameters(), lr = 0.00001)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.hybrid(x)
        
        return torch.cat((x, 1 - x), -1)

    def train_model(self, epochs, train_data_generator, len_train_images):
        loss_list = []
        self.train()
        
        for epoch in range(epochs):
            gen = iter(train_data_generator)
            total_loss = []
            for i in range(len_train_images):
                
                (data, target) = next(gen)
                self.optimizer.zero_grad()
                # Forward pass
                output = self.forward(data)
                # Calculating loss
                loss = self.loss_function(output, target)
                # Backward pass
                loss.backward()
                # Optimize the weights
                self.optimizer.step()
                
                total_loss.append(loss.item())
                print('\rImage '+str(i+1)+' of ' + str(len_train_images) + ' Loss: %.5f' % loss.item(), end='')
            
            loss_list.append(sum(total_loss)/len(total_loss))
            print(' ->  Training [{:.0f}%]\tEpoch Loss: {:.4f}'.format(100. * (epoch + 1) / epochs, loss_list[-1]))
        
        return loss_list

    def evaluate_model(self, val_data_generator, len_val_images):
        total_loss = []
        gen = iter(val_data_generator)

        predictions = []
        ground_truth = []

        self.eval()
        loss_func = self.loss_function
        with torch.no_grad():
            correct = 0
            for i in range(len_val_images):

                (data, target) = next(gen)
                output = self.forward(data)
                
                pred = output.argmax(dim=1, keepdim=True)

                predictions.append(pred.item())
                ground_truth.append(target.item())

                #correct += pred.eq(target.view_as(pred)).sum().item()

                #loss = loss_func(output, target)
                #total_loss.append(loss.item())

            #print('\n Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
            #    sum(total_loss) / len(total_loss),
            #    correct / len(val_images) * 100)
            #    )
            predictions = np.array(predictions)
            ground_truth = np.array(ground_truth)

            cm = confusion_matrix(predictions, ground_truth)

            # correct predictions / samples * 100
            accuracy = (cm[0,0] + cm[1,1]) / sum(sum(cm)) * 100
            # true_positive/true_positive+False_positive
            precision = (cm[0,0] + cm[1,1])/((cm[0,0] + cm[1,1])+cm[0,1]) * 100
            # true_positive/true_positive+False_negative
            recall = (cm[0,0] + cm[1,1])/((cm[0,0] + cm[1,1])+cm[1,0]) * 100
            # 2 * (precision * recall) /(precsion+recall)
            f1 = 2 * (precision * recall)/(precision+recall) 

            print('Accuracy: %.2f %%' % accuracy)
            print('Precion: %.2f %%' % precision)
            print('Recall: %.2f %%' % recall)
            print('F1 score: %.2f %%' % f1)

            return accuracy, precision, recall, f1