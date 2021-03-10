from tensorflow.keras.layers import Dense, Dropout, Conv2D, Activation, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential 
from tensorflow.keras.optimizers import Adam
from time import time

class CNN_Classifier:

    def __init__(self, img_shape, n_classes):
        self.img_shape = img_shape
        self.n_classes = n_classes

        self.model = self.define_model()
        self.model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

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
        
        init = time()
        history = self.model.fit(train_data_generator,
                                steps_per_epoch = train_dataset_len//batch_size,
                                epochs=epochs,
                                validation_data=val_data_generator,
                                validation_steps=val_dataset_len//batch_size,    
                                callbacks=[es],
                            )

        elapsed_time = time() - init
        print('Training time: ', elapsed_time)

        return history
