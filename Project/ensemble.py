import os
import time
import json
import argparse

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import pandas as pd
from sklearn.utils import shuffle

import models
from utils.params import Params
from utils import Datasets
#import utils.text_normalizer as tn

import random
import warnings
warnings.filterwarnings("ignore")

def main():
    start_time = time.strftime("%d%m%y_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_1", 
        type=str, 
        help="Pass the name of the first model"
        )

    parser.add_argument(
        "--model_name_2", 
        type=str, 
        help="Pass the name of the second model"
        )

    args = parser.parse_args()
  
    # Write data if specified in command line arguments. 

    Dataset = getattr(Datasets, "IMDBDataset")
    params = Params("hyper_params.yaml", 'default')
    dataset = Dataset(os.path.join(params.data_dir, 'train_data_3.csv'),
                      os.path.join(params.data_dir, 'valid_data_3.csv'),
                      os.path.join(params.data_dir, 'test_data_3.csv'), 
                      10000,
                      False)
    log_dir = './log5/ensemble'
    if not os.path.exists(log_dir): os.makedirs(log_dir)

    args = parser.parse_args()
    #training the first model 
    print('training the first model ', args.model_name_1)
    params = Params("hyper_params.yaml", args.model_name_1)
    model_module_1 = __import__('.'.join(['models', params.model_name]),  fromlist=['object'])
    model_1 = model_module_1.net(dataset, params.dropout, params.dropout2) 
    model_1.train(dataset, params.batch_size, params.log_dir, params.num_epochs)

    #training the second model 
    params = Params("hyper_params.yaml", args.model_name_2)
    print('training the second model ', args.model_name_2)
    model_module_2 = __import__('.'.join(['models', params.model_name]),  fromlist=['object'])
    model_2 = model_module_2.net(dataset, params.dropout, params.dropout2) 
    model_2.train(dataset, params.batch_size, params.log_dir, params.num_epochs)

    # Load model that has been chosen via the command line arguments. 

    print('Evaluation on Test Dataset')
    model1 = model_1.model
    model2 = model_2.model
    # test_loss, test_acc = self.model.evaluate(dataset.test_dataset[0], dataset.test_dataset[1])

    predicted_classes1 = np.concatenate(model1.predict(dataset.test_dataset[0]))
    predicted_classes2 = np.concatenate(model2.predict(dataset.test_dataset[0]))

    predicted_classes = np.where((predicted_classes1 + predicted_classes2) / 2 > 0.5, 1, 0)
    print('lengh of test set', len(predicted_classes))
    subset = predicted_classes[predicted_classes == dataset.test_dataset[1]]
    tp = subset[subset == 1]
    tn = subset[subset == 0]
    print('true positive', len(tp))
    print('true negative', len(tn))
    print('Ensemble Test accuracy: ', float(len(tn) + len(tp))/len(predicted_classes))

    

if __name__ == '__main__':
    main()
