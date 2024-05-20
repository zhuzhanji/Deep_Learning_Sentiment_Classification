import os
import time
import datetime

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

from tensorflow.keras.callbacks import TensorBoard
from models.model import model


class net(model):
    def __init__(self, data,dropout, dropout2):

        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
            input_dim= 10000,
            input_length = 500,
            output_dim=64,
            # Use masking to handle the variable sequence lengths
            mask_zero=True),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            #tf.keras.layers.LSTM(100, dropout = dropout, recurrent_dropout = recurrent_dropout),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(dropout2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
        
        super().__init__(model)
        


