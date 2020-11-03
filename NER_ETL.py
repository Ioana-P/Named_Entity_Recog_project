## principal functions and objects file

# clear sections are shown in comments
# go to docstrings for function purpose and arguments

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# import matplotlib.pyplot as plt
# import seaborn as sns

import argparse
import json
import os
import pickle
import sys
import torch
import torch.optim as optim
import torch.utils.data
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import csv



def load_and_split_data(csv_file_path :str , 
                            train_size : float, test_size : float,
                             val_size = 0.0):
        """[summary]

        Args:
            csv_file_path ([type]): [description]
        """     
        if ((train_size+test_size!=1.0) and val_size==0.0):
            val_size = 1.0 - (train_size+test_size)

        with open(csv_file_path) as f:
            csv_reader = csv.reader(f, delimiter=',', quotechar='"')
            next(csv_reader)

        data = pd.read_csv(csv_file_path, index_col=0)

        train_val , test = train_test_split(data, test_size = test_size, 
                                            train_size = (1.0 - test_size),
                                            random_state=12345, 
                                            shuffle=False, )

        test.to_csv('clean_data/TEST_clean_data.csv')

        if val_size!=0.0:
            train, valid = train_test_split(train_val, test_size = val_size,
                                            random_state=12345, shuffle=False)
            train.to_csv('clean_data/TRAIN_clean_data.csv')
            valid.to_csv('clean_data/VALID_clean_data.csv')
        else:
            train_val.to_csv('clean_data/TRAIN_clean_data.csv')

        return


#################################CLEANING#####################################


#################################ETL#####################################
class EntityETL(object):
    """Class for extracting, transforming and loading the prepped text data 
    for Named Entity Recognition (NER) - made to interface with NER model
    in training, prediction and out-of-sample use. 

    Args:
        embedding file (csv filepath): if you have a file with pretrained 
        embeddings (e.g. Glove 50 dim), load it here; will be passed to NER 
        model
        embedding_dim (int): dimension of embedding, e.g. 50
        lang (str): language of text, default english 'en'
        
    """
    def __init__(self, embedding_file = None, embedding_dim = 0, lang = 'en'):
        """Instantiating object for extracting, transforming and loading the prepped text data 
        for Named Entity Recognition (NER) - made to interface with NER model
        in training, prediction and out-of-sample use. 

        Args:
            embedding file (csv filepath): if you have a file with pretrained 
            embeddings (e.g. Glove 50 dim), load it here; will be passed to NER 
            model
            embedding_dim (int): dimension of embedding, e.g. 50
            lang (str): language of text, default english 'en'
        
        """
        self.embedding_file = embedding_file
        self.embedding_dim = embedding_dim
        self.lang = lang
        self.vocab = {}
        self.vocab_size = len(self.vocab)
        self.embed_dict={}
        self.batch_starting_point = 0

        return


    def load_train_vocab_nn(self, csv_file_path ):
        """Method for loading input data for model VOCABULARY for neural model

        Args:
            csv_file_path (str): location of csv file
        """        
        #overwrites vocab as empty dict
        self.vocab = {}
        self.ne_tag_map = {}
        self.pos_tag_map = {}

        self.vocab['UNK'] = 1
        self.ne_tag_map['UNK_NE'] = 1
        self.pos_tag_map['UNK_POS'] = 1

        self.vocab['PAD'] = 0
        self.ne_tag_map['PAD'] = 0
        self.pos_tag_map['PAD'] = 0

        list_ne_tags = []
        list_pos_tags = []
        list_words = []
        with open(csv_file_path) as f:
            csv_reader = csv.reader(f, delimiter=',', quotechar='"')
            next(csv_reader)
            # for line in f.read().splitlines()[1:]:
            for line in csv_reader:
                # print(line)
                list_words.append(line[2])
                try:
                    list_ne_tags.append(line[4])
                    list_pos_tags.append(line[3])
                except:
                    list_ne_tags.append('O')
                    list_pos_tags.append('.')
                    print(line)
                    continue

        set_words = set(list_words)
        set_ne_tags = set(list_ne_tags)
        set_pos_tags = set(list_pos_tags)

        for i, word in enumerate(set_words, start=2):
            self.vocab[word] = i
        for i, ne_tag in enumerate(set_ne_tags, start=2):
            self.ne_tag_map[ne_tag] = i
        for i, pos_tag in enumerate(set_pos_tags, start=2):
            self.pos_tag_map[pos_tag] = i
        self.vocab_size = len(self.vocab)
        return 

    def load_input_data(self, csv_file_path):
        """Loads the input training data into the object from a file
        for model training purposes. At this point out is still roughly 
        compatible with either feature-based or neural model.

        Args:
            csv_file_path (str): location of csv file containing training 
            data
            split_by (str) : separator to be used for 

        Returns:
            tuple: train_sentences (list of lists of words, each sublist a sentence);
                   train_labels (same format as above)
        """        
        self.train_sentences = []
        self.train_labels = []
        self.train_pos_tags = []
        with open(csv_file_path) as f:
            csv_reader = csv.reader(f, delimiter=',', quotechar='"')
            next(csv_reader)
            # for line in f.read().splitlines()[1:]:
            
            s_index='1.0'
            sentence=[]
            sent_labels = []
            sent_pos_tags = []
            for line in csv_reader:
        #         print(line)
                
                if s_index != line[1]:
                    s_index = line[1]
                
                    self.train_sentences.append(sentence)
                    self.train_labels.append(sent_labels)
                    self.train_pos_tags.append(sent_pos_tags)
                    sentence = []
                    sent_labels = []
                    sent_pos_tags = []
                    
                    w_index = line[0]
                    # s_index = line[1]
                    word = line[2]
                    pos_tag = line[3]
                    label = line[4]
                    sentence.append(word)
                    sent_labels.append(label)
                    sent_pos_tags.append(pos_tag)
                    continue
                    
                w_index = line[0]
                s_index = line[1]
                word = line[2]
                pos_tag = line[3]
                label = line[4]
                
                
            #           s = [self.vocab[token] if token in self.vocab 
            #               else vocab['UNK'] for token in sentence.split(' ')]
                sentence.append(word)
                sent_labels.append(label)
                sent_pos_tags.append(pos_tag)
                
        return self.train_sentences, self.train_labels


    def load_embed_vects(self, embedding_file = None, embedding_dim = None, return_array=False):
        """Loads and stores embedding vectors. Call function with empty params 
        to simply use the prespecified embedding_file and dim set at 
        instantiation.

        Args:
            embedding file (csv filepath): if you have a file with pretrained 
            embeddings (e.g. Glove 50 dim), load it here; will be passed to NER 
            model
            embedding_dim (int): dimension of embedding, e.g. 50
        """        
        if embedding_file!=None:
            self.embedding_file=embedding_file
        if embedding_dim!=None:
            self.embedding_dim=embedding_dim

        avg_vect = np.zeros((self.embedding_dim))
    
        with open(self.embedding_file, 'rb') as f:

            for line in f:
                parts = line.split()
                word = parts[0].decode('utf-8')
                vector = np.array(parts[1:], dtype=np.float32)
                self.embed_dict[word] = vector
                avg_vect += vector
            # creating the vector for new, UNKnown words in the vocabulary
            # NOTE, this is NOT the same as the word "unk", which is 
            # present in GloVe's vocabulary, for instance
            self.embed_dict['UNK'] = avg_vect/len(self.embed_dict)
            self.embed_dict['PAD'] = np.zeros((self.embedding_dim))

            self.embedding_weights_matrix = np.zeros(((len(self.embed_dict)+2) ,  embedding_dim))

            self.embedding_weights_matrix[0] = self.embed_dict['PAD']
            self.embedding_weights_matrix[1] = self.embed_dict['UNK']
            for i, word in enumerate(self.embed_dict.keys(), start=2):
                self.embedding_weights_matrix[i] = self.embed_dict[word]

        return
        
    def prep_input_for_nn(self, train_sentences= None, train_labels = None,
                        train_pos_tags = None):
        """[summary]

        Args:
            train_sentences (list): 
            train_labels (list): [description]
            train_pos_tags (list)
        Returns:
            tuple : nn_train_sentences, nn_train_pos (optional), nn_train_sentences 
            in the format for neural models 
        """              
        if train_sentences!=None:
            self.train_sentences = train_sentences
        if train_labels!=None:
            self.train_labels = train_labels
        if train_pos_tags!=None:
            self.train_pos_tags = train_pos_tags

        nn_train_sentences = []
        nn_train_labels = []
        nn_train_pos = []

        for sentence in self.train_sentences:
            sent = [self.vocab[token] if token in self.vocab.keys() else vocab['UNK'] 
                    for token in sentence]
            nn_train_sentences.append(sent)

        for label_sent in self.train_labels:
            labels = [self.ne_tag_map[label] if label in self.ne_tag_map else ne_tag_map['UNK_NE'] 
                      for label in label_sent]
            nn_train_labels.append(labels)

        for pos_sent in self.train_pos_tags:
            pos = [self.pos_tag_map[pos] if pos in self.pos_tag_map else self.pos_tag_map['UNK_POS']
                   for pos in pos_sent]
            nn_train_pos.append(pos)

        if train_pos_tags == None :
            return nn_train_sentences, nn_train_labels
        else:
            return nn_train_sentences, nn_train_pos, nn_train_labels

    def nn_batch_generator(self, train_sentences_nn, train_labels_nn, train_pos_tags_nn = None, batch_len = 50):
        """Performs preprocessing and tokenisation steps for neural model.

        Args:
            csv_file_path ([type]): [description]

        Returns:
            [type]: [description]
        """   

        #compute length of longest sentence in batch
        # for b_ind in range(batch_starting_point, batch_starting_point + 50):s
        try:
            batch_sentences = train_sentences_nn[self.batch_starting_point:self.batch_starting_point+ batch_len]
            batch_labels = train_labels_nn[self.batch_starting_point:self.batch_starting_point+ batch_len]
        except IndexError:
            self.batch_starting_point = 0
        
        # print(len(batch_sentences[0]))
        # print(len(batch_labels[0]))
        # POS tag implementation stopping here until further notice -- WILL COME BACK TO THIS!
        # batch_pos_tags = train_pos_tags_nn[:self.batch_starting_point+ batch_len]
        while True:
            batch_max_sent_len = max([len(s) for s in batch_sentences])

            #   BIG QUESTION HERE: WHY WOULD WE ONLY PAD UP TO LENGTH OF MAX SENTENCE IN A SINGLE BATCH??
            # SURELY WE'LL HAVE TO DO THAT FOR THE MAX SENTENCE OF THE ENTIRE TRAINING SET INSTEAD
            #  Maybe I'm missing something here, in which case I'll leave it as it is ftm
            #  note to self: if fixing, move the np.array sizing outside of the main loop
            batch_data = self.vocab['PAD']*np.ones((len(batch_sentences), batch_max_sent_len))
            batch_ne_tags = self.ne_tag_map['PAD']*np.ones((len(batch_sentences), batch_max_sent_len))

            for j in range(len(batch_sentences)):
                cur_len = len(batch_sentences[j])
                batch_data[j][:cur_len] = batch_sentences[j]
                # print(cur_len)
                # print(len(batch_labels[j]))
                
                # print(batch_labels[j])
                batch_ne_tags[j][:cur_len] = batch_labels[j]
            # print(batch_data)
            batch_data, batch_ne_tags = torch.LongTensor(batch_data), torch.LongTensor(batch_ne_tags)
            # batch_data, batch_ne_tags = torch.Tensor(batch_data), torch.LongTensor(batch_ne_tags)

            batch_data, batch_ne_tags = Variable(batch_data), Variable(batch_ne_tags)

            self.batch_starting_point+=batch_len
            yield batch_data, batch_ne_tags

    
       



#################################DATA TRANSFORMATION#####################################


def load_glove_vects(file = 'glove/glove.6B.50d.txt', vdim=None):
    """Function that loads the Global representation Vectors
    and returns them as a dictionary. 
    -----------------
    Returns:
    glove_dict - (dict) key - word (str), value - n-dimensional np array """
    glove_dict = {}
#     total_vocab = vocab
    if type(vdim)==int:
        file = f'glove/glove.6B.{vdim}d.txt'
    avg_vect = np.zeros((vdim,))
    with open(file, 'rb') as f:
        for line in f:
            parts = line.split()
            word = parts[0].decode('utf-8')
            vector = np.array(parts[1:], dtype=np.float32)
            glove_dict[word] = vector
            avg_vect += vector
        # creating the vector for new, UNKnown words in the vocabulary
        # NOTE, this is NOT the same as the word "unk", which is 
        # present in glove's vocabulary
        glove_dict['UNK'] = avg_vect/len(glove_dict)
        glove_dict['PAD'] = np.zeros((vdim,))
    return glove_dict

def generate_tag_set(targets_list : list):
    """Function takes in a list of NE tags (which should include at least one instance of 
    every possible NE tag) and returns a dict matching each NE tag to a unique int.
    Returns:
    tag_map - (dict) of NE tag - associated int pairs"""
    ne_dict = {}
    i = 0
    for sublist in targets_list:
        for ne in sublist:
            if ne in ne_dict.keys():
                continue
            else:
                ne_dict[ne] = i
                i += 1
    return ne_dict



def sent_to_vect(feature_list : list, targets_list : list, vocab : dict, ne_dict : dict):
    """Function takes in list of lists of dictionaries (input data), target NE labels,
    a vocabulary (dictionary) and a dict of NE tags and their corresponding identifiers;
    Returns a list of vectorised input data"""
    vect_sentences = []        
    vect_label_sentences = []
    
    for sentence in feature_list:     
        #replace each token by its index if it is in vocab
        #else use index of UNK
        sentence_vectors = [vocab[token[0]] if token[0] in vocab.keys() 
             else vocab['UNK']
             for token in sentence]
        vect_sentences.append(sentence_vectors)
        
    for sentence in targets_list:
        #replace each label by its index
        try:
            label_sent = [ne_dict[label] for label in sentence]
        except:
            generate_tag_set(targets_list)
            label_sent = [ne_dict[label] for label in sentence]
        vect_label_sentences.append(label_sent) 
        
    return vect_sentences, vect_label_sentences

def prep_batch(batch_sentences : list, batch_sentences_labels : list, vocab : dict, word_vect_dim = 50):
    """Function takes in a list of lists (each sublist a sentence of n-dimension
    numpy arrays), the associated list of lists of NE labels and a vocabulary (dict)"""
    #compute length of longest sentence in batch
    batch_max_len = max([len(sentence) for sentence in batch_sentences_labels])
    #prepare a numpy array with the data, initializing the data with 'PAD' 
    #and all labels with -1; initializing labels to -1 differentiates tokens 
    #with tags from 'PAD' tokens
    #note the dimensional change here as we are effectively about to 
    # concatenate the sentences along the 2nd dimension
    batch_data = np.zeros((len(batch_sentences), batch_max_len, word_vect_dim))
    batch_labels = -1*np.ones((len(batch_sentences), batch_max_len))
    #copy the data to the numpy array
    for j in range(len(batch_sentences)):
        #accessing individual sentence below
        cur_len = len(batch_sentences[j])
        
        for k in range(len(batch_sentences[j])):
            #accessing individual word vectors below
            batch_data[j,k, :] = batch_sentences[j][k].reshape(1,-1)
            
        batch_labels[j][:cur_len] = batch_sentences_labels[j]

    #since all data are indices, we convert them to torch LongTensors
    batch_data, batch_labels = torch.Tensor(batch_data), torch.Tensor(batch_labels)

    #convert Tensors to Variables
    # Torch tensors and torch Variables are almost the same, the latter being a wrapper fn
    # that allows for additional methods to be called onto the underlying tensor. 
    # So we're reassigning them as Variables for extra future flexibility
#     batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)
    return batch_data, batch_labels
    

#################################EDA#####################################

def example_output(num, data, estimator, target_data, word_arg = 'word.lower()'):
    '''Short function that produces a quick inspection table so we can compare how 
    our CRF model performs against the labelled sentences. 
    Params:
    num - (int) index of sentence in the data
    data - (list) input data that the prediction will be made on. Must be of form of list
    of lists of dicts
    estimator - (object) model to use for prediction - has to be sklearn crf
    target_data - (list) data from which ground truth labels are to be retrieved; has
    to be list of list of strings
    word_arg - (str) which dictionary argument to access to retrieve the original, 
    lowercased word - this will depend on how the initial data was defined. 
    
    Returns:
    pandas Dataframe with 3 columns: True label; predicted label and the Word
    '''
    predicted_table = pd.DataFrame(columns=['True', 'Pred', 'Word'])
    predicted_table['True'] = target_data[num]
    predicted_table['Word'] = [word[word_arg] for word in data[num]]
    predicted_table['Pred'] = estimator.predict([data[num]])[0]
    return predicted_table

    


#################################SUMMARY TABLES CREATION#####################################



#############################MODEL BUILDING, GRIDSEARCH AND PIPELINES#####################################

class LSTMClassifier(nn.Module):
    """
    This is the simplest RNN model we will be using to perform NER.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.softm = nn.Softmax()
        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        x = x.t()
        reviews = x[1:,:]
        embeds = self.embedding(reviews)
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(lstm_out)
        out = out[lengths - 1, range(len(lengths))]
        return self.softm(out.squeeze())


#############################MODEL EVALUATION (METZ, ROC CURVE, CONF_MAT)#####################################

