
# coding: utf-8

# ** MNIST CNN CLassifier Using Keras**

# In[ ]:


import sys
import os


# In[ ]:


import numpy as np


# In[ ]:


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

class MNIST_K_CNN(object):
    @staticmethod
    def larger_model():
        # create model
        model = Sequential()
        model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(15, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def __init__(self, epochs=10, batch_size=200, validation_split=.3, verbose=0):
        self._epochs = epochs
        self._batch_size = batch_size
        self._validation_split = validation_split
        self._verbose = verbose

    def fit(self, X, y, epochs=None, batch_size=None, 
            validation_split=None, verbose=None):
        self._model = self.larger_model()
        X = X.reshape(X.shape[0], 1, 28, 28)
        y = np_utils.to_categorical(y)
        if epochs is None:
            epochs = self._epochs
        if batch_size is None:
            batch_size = self._batch_size
        if validation_split is None:
            validation_split = self._validation_split
        if verbose is None:
            verbose = self._verbose
        self._model.fit(X, y, validation_split=validation_split, epochs=epochs, verbose=verbose)
        
    def predict(self, X):
        X = X.reshape(X.shape[0], 1, 28, 28)
        y_pred = self._model.predict(X)
        return np.argmax(y_pred, axis=1)


# In[ ]:


if __name__ == '__main__':
    from mnist_adv_test import load_data
    from sklearn.metrics import accuracy_score
    x_train, y_train, x_test, y_test, _ = load_data(use_p=1.)
    
    k_cnn = MNIST_K_CNN()
    k_cnn.fit(x_train, y_train, verbose=1)
    y_pred = k_cnn.predict(x_test)
    print("Accuracy={}".format(accuracy_score(y_test, y_pred)))

