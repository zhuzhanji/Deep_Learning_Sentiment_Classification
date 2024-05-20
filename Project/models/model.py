import os
import time
import datetime

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.callbacks import TensorBoard

from numpy import savetxt

class model():
    def __init__(self, model):
        self.model = model
        model.summary()


    def train(self, dataset, batch_size, dstdir, epoch):
        mdir = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
        logdir = dstdir + '/' + mdir
        print('Training Log saving to', logdir)
        tbCallBack = TensorBoard(log_dir=logdir, write_graph=True)

        checkpoint_path = dstdir + "/model/weights.h5"
        print('Weights saving to', checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                monitor='val_accuracy',
                                                mode='max',
                                                save_weights_only=True,
                                                save_best_only=True)

        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)                                    
        
        history = self.model.fit(                    
                    dataset.train_dataset[0], 
                    dataset.train_dataset[1], 
                    epochs= epoch,
                    validation_data=dataset.valid_dataset,
                    validation_steps=30,
                    batch_size = batch_size,
                    callbacks = [cp_callback, tbCallBack, es_callback])

        print('Loading best model')
        self.model.load_weights(checkpoint_path)
        #print('Best model saving to ')
        tf.keras.models.save_model(self.model, dstdir + "/model/model.h5")

        print('Evaluation on Test Dataset')
        # test_loss, test_acc = self.model.evaluate(dataset.test_dataset[0], dataset.test_dataset[1])

        predicted_classes = np.concatenate(np.where(self.model.predict(dataset.test_dataset[0]) > 0.5, 1, 0))
        print('lengh of test set', len(predicted_classes))
        subset = predicted_classes[predicted_classes == dataset.test_dataset[1]]

        pred = predicted_classes == dataset.test_dataset[1]
        savetxt('./prediction.csv', pred, delimiter=',')

        tp = subset[subset == 1]
        tn = subset[subset == 0]
        print('true positive', len(tp))
        print('true negative', len(tn))
        print('Test accuracy: ', float(len(tn) + len(tp))/len(predicted_classes))

