## principal functions and objects file

# clear sections are shown in comments
# go to docstrings for function purpose and arguments

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as shc

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from sklearn.linear_model import LogisticRegression
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from matplotlib import cm
import numpy as np
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import silhouette_score
import string

from collections import Counter
import scipy.stats as ss
from dython.nominal import conditional_entropy, associations

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
tokenizer = RegexpTokenizer(r'\b\w{3,}\b')
stop_words = list(set(stopwords.words("english")))
stop_words += list(string.punctuation)
# stop_words += ['__', '___']

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')






#################################CLEANING#####################################





#################################DATA TRANSFORMATION#####################################
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


#############################MODEL EVALUATION (METZ, ROC CURVE, CONF_MAT)#####################################

