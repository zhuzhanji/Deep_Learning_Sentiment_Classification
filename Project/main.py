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
        "model_name", 
        type=str, 
        help="Pass name of model as defined in hyper_params.yaml."
        )
    parser.add_argument(
        "--write_data",
        required = False,
        type = int,
        default=False,
                help="Set to true to write_data."
        )
    args = parser.parse_args()
    # Parse our YAML file which has our model parameters. 
    params = Params("hyper_params.yaml", args.model_name)

    # Write data if specified in command line arguments. 
    if args.write_data == 1:
        print('--- Preprocessing Data ---')
        data = pd.read_csv(os.path.join(params.data_dir, params.data_file))
        rows = len(data)
        for i in range(rows):
          if i % 10000 == 0:
            print('processing row', i)
          data.loc[i, 'review'] = data.iloc[i]['review'].replace('<br />', ' ')
          #data.loc[i, 'review'] = tn.normalize_corpus(data.iloc[i]['review'])
          data.loc[i, 'sentiment'] = (data.iloc[i]['sentiment'] == 'positive') * 1
        random.seed(2023)
        index = list(range(len(data)))
        train_split = round(data.shape[0]*0.64)
        valid_split = round(data.shape[0]*0.8)
        train_data = data.iloc[index[:train_split]]
        valid_data = data.iloc[index[train_split:valid_split]]
        test_data = data.iloc[index[valid_split:]]
        train_data.to_csv(os.path.join(params.data_dir, 'train_data_3.csv'), index=False)
        valid_data.to_csv(os.path.join(params.data_dir, 'valid_data_3.csv'), index=False)
        test_data.to_csv(os.path.join(params.data_dir, 'test_data_3.csv'), index=False)
        print('--- Preprocessing Data Finished---')
        #reset random seed
        random.seed(None)

    Dataset = getattr(Datasets, params.dataset_class)
    dataset = Dataset(os.path.join(params.data_dir, 'train_data_3.csv'),
                      os.path.join(params.data_dir, 'valid_data_3.csv'),
                      os.path.join(params.data_dir, 'test_data_3.csv'), 
                      params.vocab_size,
                      params.pretrain)

    if not os.path.exists(params.log_dir): os.makedirs(params.log_dir)

    # Load model that has been chosen via the command line arguments. 
    model_module = __import__('.'.join(['models', params.model_name]),  fromlist=['object'])
    model = model_module.net(dataset, params.dropout, params.dropout2) 
    model.train(dataset, params.batch_size, params.log_dir, params.num_epochs)



if __name__ == '__main__':
    main()
