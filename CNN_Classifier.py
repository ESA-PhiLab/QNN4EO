from tensorflow.keras.layers import Dense, Dropout, Conv2D, Activation, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential 
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from utils import reverseOneHotEncoding
from time import time
import numpy as np

class CNN_Classifier:

    def __init__(self, img_shape, n_classes):
        self.img_shape = img_shape
        self.n_classes = n_classes

        self.model = self.define_model()
        self.model.compile(optimizer=Adam(lr=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

    def define_model(self):
        kernel_size = 5 
        stride_size = 1 
        pool_size = 2   

        model = Sequential()
        
        # Convolutional Layer 1
        model.add(Conv2D(3, kernel_size, strides=stride_size, padding='same', activation='relu', input_shape=self.img_shape))
        model.add(MaxPool2D(pool_size=pool_size, strides=(2, 2)))
        # Convolutional Layer 2
        model.add(Conv2D(6, kernel_size, strides=stride_size, activation='relu'))
        model.add(MaxPool2D(pool_size=pool_size, strides=(2, 2)))
        # Convolutional Layer 3
        model.add(Conv2D(16, kernel_size, strides=stride_size, activation='relu'))
        model.add(MaxPool2D(pool_size=pool_size, strides=(2, 2)))
        
        model.add(Flatten())

        # Dense Layer 1
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
         # Dense Layer 2
        model.add(Dense(self.n_classes, activation='sigmoid'))

        return model

    def train_model(self,  epochs, batch_size, train_data_generator, val_data_generator, train_dataset_len, val_dataset_len):
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
        
        #init = time()
        history = self.model.fit(train_data_generator,
                                steps_per_epoch = train_dataset_len//batch_size,
                                epochs=epochs,
                                validation_data=val_data_generator,
                                validation_steps=val_dataset_len//batch_size,    
                                callbacks=[es],
                            )

        #elapsed_time = time() - init
        #print('Training time: ', elapsed_time)

        return history

    def evaluate_model(self, val_data_generator, classes_name):
        x_train, y_train = next(iter(val_data_generator))
        pred = self.model.predict(x_train)

        pl = reverseOneHotEncoding(pred, classes_name)
        cl = reverseOneHotEncoding(y_train, classes_name)

        cm = confusion_matrix(pl, cl)

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