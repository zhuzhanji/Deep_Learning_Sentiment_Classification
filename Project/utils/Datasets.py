import os
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
import numpy as np
import pandas as pd

class IMDBDataset():
    def __init__(self, train_path, valid_path, test_path, VOCAB_SIZE, pretrain):
        """
        train_path (str): Path to training dataset. 
        valid_path (str): Path to valid dataset.
        test_path (str): Path to test dataset.
        VOCAB_SIZE (int): size of dictionary to be built
        pretrain (bool): whether using pretrained glove model
        """
        #Loading trainning, validation and test set
        train_data = pd.read_csv(train_path)
        valid_data = pd.read_csv(valid_path)
        test_data = pd.read_csv(test_path)
        print('length of train_data', len(train_data))
        print('length of valid_data', len(valid_data))
        print('length of test_data', len(test_data))

        train_features = train_data.copy()
        train_labels = train_features.pop('sentiment')
        

        valid_features = valid_data.copy()
        valid_labels = valid_features.pop('sentiment')
        

        test_features = test_data.copy()
        test_labels = test_features.pop('sentiment')


        # Using training set to build dictionary
        # Training, validation and test set are vectorised.
        if not pretrain:

          np_train_features = np.concatenate(np.array(train_features))
          np_train_labels = np.array(train_labels)
          np_valid_features = np.concatenate(np.array(valid_features))
          np_valid_labels = np.array(valid_labels)
          np_test_features = np.concatenate(np.array(test_features))
          np_test_labels = np.array(test_labels)
          
          print('TextVectorization Layer')
          encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE,ragged = True)
          encoder.adapt(np_train_features)

          np_train_features = encoder(np_train_features).numpy()
          np_train_features = sequence.pad_sequences(np_train_features, maxlen=500, padding='post')

          np_valid_features = encoder(np_valid_features).numpy()
          np_valid_features = sequence.pad_sequences(np_valid_features, maxlen=500, padding='post')

          np_test_features = encoder(np_test_features).numpy()
          np_test_features = sequence.pad_sequences(np_test_features, maxlen=500, padding='post')

          self.train_dataset = (np_train_features, np_train_labels)
          self.valid_dataset = (np_valid_features, np_valid_labels)
          self.test_dataset = (np_test_features, np_test_labels)

        # Using Glove pre-trained model
        else:
          vectorizer, self.embedding_dim, self.embedding_matrix = self.getImbedding(train_features)
          x_train = vectorizer(np.array(train_features)).numpy()
          x_val = vectorizer(np.array(valid_features)).numpy()
          x_test = vectorizer(np.array(test_features)).numpy()
          
          y_train = np.array(train_labels)
          y_val = np.array(valid_labels)
          y_test = np.array(test_labels)

          self.train_dataset = (x_train, y_train)
          self.valid_dataset = (x_val, y_val)
          self.test_dataset = (x_test, y_test)



    # This implementation references https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    def getImbedding(self, train_features):
      vectorizer = tf.keras.layers.TextVectorization(max_tokens=10000 - 2, output_sequence_length=500)
      vectorizer.adapt(np.array(train_features['review']))

      voc = vectorizer.get_vocabulary()
      word_index = dict(zip(voc, range(len(voc))))

      path_to_glove_file = "./data/glove.6B.50d.txt"

      embeddings_index = {}
      with open(path_to_glove_file) as f:
          for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

      print("Found %s word vectors." % len(embeddings_index))

      num_tokens = len(voc) + 2
      embedding_dim = 50
      hits = 0
      misses = 0

      # Prepare embedding matrix
      embedding_matrix = np.zeros((num_tokens, embedding_dim))
      for word, i in word_index.items():
          embedding_vector = embeddings_index.get(word)
          if embedding_vector is not None:
              # Words not found in embedding index will be all-zeros.
              # This includes the representation for "padding" and "OOV"
              embedding_matrix[i] = embedding_vector
              hits += 1
          else:
              misses += 1
      print("Converted %d words (%d misses)" % (hits, misses))

      return vectorizer,embedding_dim, embedding_matrix



