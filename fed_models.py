import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from requests import get
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, metrics
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Reshape, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2



# put your federated models here. Called by server and client
def get_model():
    # m = AutoencoderFlat(n_features=784, hidden_neurons=[16, 8, 16])
    # m = AutoencoderConv() 
    m = AutoencoderConvV2() 
    m.build(input_shape=(None,784))
    m.compile(optimizer='adam', loss='mse')
    return m
    


class AutoencoderFlat(Model): 
    def __init__(self, n_features, dropout_rate=0.1, hidden_neurons=[64, 32, 64]):
        super(AutoencoderFlat, self).__init__()
        #self.latent_dim = latent_dim

        self.n_features = n_features
        self.hidden_neurons = hidden_neurons
        self.dropout_rate = dropout_rate
        self.hidden_neurons = hidden_neurons

        first = [
            # tf.keras.Input(shape=(n_features,)),
            # layers.Dropout(self.dropout_rate) # to add dropout
        ]
        internal = []

        for neu in self.hidden_neurons:
            internal.append(layers.Dense(neu, activation='relu'))
            #internal.append(layers.Dropout(self.dropout_rate))

        output = [
            layers.Dense(n_features, activation='sigmoid')
            #layers.Reshape((28, 28))
        ]
        self.mod = tf.keras.Sequential(first+internal+output)

    def call(self, x):
        mod = self.mod(x)
        return mod

# hardcoded to 28x28 images
class AutoencoderConv(Model):
    def __init__(self):
        super(AutoencoderConv, self).__init__()
        
        self.encoder = tf.keras.Sequential([
            layers.Reshape((28, 28, 1)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)
            ])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same'),
            layers.Flatten()
            ])
        
            

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# hardcoded to 28x28 images
class AutoencoderConvV2(Model): 
    def __init__(self):
        super(AutoencoderConvV2, self).__init__()

        self.model = Sequential([
            Reshape((28, 28,1)),

            # Encoder
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2), padding="same"),

            # Decoder
            Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same"),
            Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same"),
            Conv2D(1, (3, 3), activation="sigmoid", padding="same"),
            Flatten()

    	])

    def call(self, x):
        model = self.model(x)
        return model


