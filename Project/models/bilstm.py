import os
import time
import datetime

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

from tensorflow.keras.callbacks import TensorBoard
from models.model import model


class net(model):
    def __init__(self, data, dropout, dropout2):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
            input_dim= 10000,
            input_length = 500,
            output_dim=64,
            # Use masking to handle the variable sequence lengths
             mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout = dropout, recurrent_dropout = dropout2)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
        
        super().__init__(model)
