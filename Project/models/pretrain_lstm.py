import os
import time
import datetime

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

from tensorflow.keras.callbacks import TensorBoard
from models.model import model


class net(model):
    def __init__(self, data, dropout, recurrent_dropout):
        num_tokens = 10000
        embedding_dim = data.embedding_dim
        embedding_matrix = data.embedding_matrix

        model = tf.keras.Sequential([
          tf.keras.layers.Embedding(num_tokens, embedding_dim,weights=[embedding_matrix],trainable=True),
          tf.keras.layers.LSTM(64, dropout = dropout, recurrent_dropout = recurrent_dropout),
          tf.keras.layers.Dense(64, activation='relu'),
          tf.keras.layers.Dense(1)
        ])
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

        super().__init__(model)
        
