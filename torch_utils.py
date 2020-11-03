import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as shc

import matplotlib.pyplot as plt
import seaborn as sns

def df_to_torch_list(df):
    """Function takes in dataframe with four columns:
    Sentence #; Word; POS; Tag.
    -------------------------------------------------
    Returns: 
    - input_data as a list of lists (each a sentence) of tuples
    where each tuple is (word; POS)
    - target_data - list of lists (each a sentence) of Named 
    Entity Tags (e.g. 'O', 'B-geo', 'I-art', etc)
    """
    
    input_data = []
    target_data = []
    data = df.copy()
    for sent_ind in range(1,len(data['Sentence #'].unique().astype(int))):
        sent_df = data.loc[data['Sentence #'] == sent_ind]
        sent_lst = []
        sent_target_lst = []
        for row in sent_df.values:
            sent_lst.append((row[1], row[2]))
            sent_target_lst.append(row[3])
        input_data.append(sent_lst)
        target_data.append(sent_target_lst)
    return input_data, target_data
        
def generate_int_vocab(input_data : list):
    """Function takes in list (corpus) of lists (sentences) of dicts (words) of tuples (word, POS) and returns
    a single vocabulary dict.
    Returns:
    vocab_dict - (dict) word - unique integer pairs."""
    vocab_dict = {}
    i=1
    for sentence in input_data:
        for word_pos_tuple in sentence:
            if word_pos_tuple not in vocab_dict.keys():
                vocab_dict[word_pos_tuple[0]] = i
                i +=1
                continue
            else: 
                continue
        vocab_dict['UNK'] = i+1
        vocab_dict['PAD'] = 0
    return vocab_dict

def generate_tag_dict(targets_list):
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

def sent_to_ints(feature_list : list, targets_list : list, vocab : dict, ne_dict : dict, incl_POS = False, POS_dict = None):
    """Function takes in list (corpus) of lists (sentences) of dicts (words) of tuples (word, POS) and returns
    a list (corpus) of lists (sentences) of integers (representing words).
    Returns:
    list_data."""
    int_sentences = []        
    int_label_sentences = []
    
    for sentence in feature_list:     
        #replace each token by its index if it is in vocab
        #else use index of UNK
        sentence_ints = [vocab[token[0].lower()] if token[0].lower() in vocab.keys() 
             else vocab['UNK']
             for token in sentence]
        int_sentences.append(sentence_ints)
        
    for sentence in targets_list:
        #replace each label by its index
        label_sent = [ne_dict[label] for label in sentence]
        int_label_sentences.append(label_sent) 
        
        
    if incl_POS:
        int_sentences_POS = []
        for sentence in feature_list:     
        #replace each token by its index if it is in vocab
        #else use index of UNK
            sentence_POS_int = [vocab[token[1]] if token[1] in POS_dict.keys() 
                 else POS_dict['UNK_POS']
                 for token in sentence]
            int_sentences_POS.append(sentence_POS_int)
        return int_sentences, int_sentences_POS, int_label_sentences
        
    else:     
        return int_sentences, int_label_sentences