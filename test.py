import tensorflow as tf
from tensorflow.keras import layers , models
import numpy as np
import matplotlib as plt 

def build_cnn_CFD_model(input_shape(64,64,2)):
    
    inputs = layers.Input(shape = input_shape)

    # (Encoder)

    x = layers.Conv2D(32,(3,3),activation='relu',padding='same')(inputs)
    x = layers.MaxPooling2D((2,2),padding = 'same' )(x)
    
    x = layers.Conv2D(64,(3,3),activation='relu',padding='same')(inputs)
    x = layers.MaxPooling2D((2,2),padding = 'same' )(x)

    x = layers.Conv2D(128,(3,3),activation='relu',padding='same')(inputs)
    encoder_output = layers.MaxPooling2D((2,2),padding = 'same' )(x)

    # (Decoder)

    x = layers.Conv2DTranspose(128,(3,3),activation='relu',padding='same')(encoder_output)
    x = layers.UpSampling(2,2)(x)

    x = layers.Conv2DTranspose(64,(3,3),activation='relu',padding='same')(x)
    x = layers.UpSampling(2,2)(x)

    x = layers.Conv2DTranspose(32,(3,3),activation='relu',padding='same')(x)
    x = layers.UpSampling(2,2)(x)

    outputs = layers.Conv2D(1,(3,3),activation='linear',padding='same')(x)
    model = models.Model(inputs = inputs,outputs = outputs )

    model.compile(optimizer='adam',loss='mse')

    return models 


def generate_mock_cfd_data(num_of_sample = 100, num_of_grid = 64):
    

    x = np.zeros(num_of_sample, num_of_grid , num_of_grid , 2)
    y = np.zeros(num_of_sample, num_of_grid , num_of_grid , 1)






































