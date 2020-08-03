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

def binarise_tags(df, col : str, negative_class_val='O'):
    """Takes in dataframe, multiclass target column and the value assoc with the negative class and adds a column 
    where the target col is binary. 
    Params:
    df - pandas dataframe
    col - (str) target column
    negative_class_val - (int/str/float) what is the value of the negative class; default : 'O'
    ------------------------
    Returns Pandas Dataframe"""
    df['binary_target'] = 0
    df.loc[df[col] != negative_class_val, 'binary_target'] = 1
    return df

def add_new_feat_df(df, output_col_names : list, return_as_df = True):
    """DATAFRAME VERSION USED FOR NON-SEQUENTIAL MODELS (e.g. Random Forests)
    FOR CRF IMPLEMENTATION, USE word_to_crf_features()
    Takes in dataframe, name of target column (e.g. 'Text'), and the name of the new column to be created.
    Assumes the order of columsn is as follows:
    0 - sentence int
    1 - text
    2 - POS tag
    3 - NE tag
    4 - binarised NE tag
    Params:
    df - pandas dataframe
    output_col_names - (str) name of the new columns of the final dataframe
    return_as_df - (bool) whethe the output should be a pandas dataframe (default True) or, for the purposes
    of computational speed, simply be a numpy array
    ---------------
    Returns pandas dataframe OR numpy array"""
    
    data = df.copy().values
    
    assert ((return_as_df) & (output_col_names!=None)), "Column names error: if you want a dataframe you must supply a list of column names"
    
    
    new_feat = np.zeros((data.shape[0],(len(output_col_names)- len(df.columns))), dtype='O')
    for row_i in range(len(data)):
        new_feat[row_i, 0] = int(data[row_i, 1].istitle()) #is It Capitalised Like A Title?
        new_feat[row_i, 1] = len(data[row_i, 1]) # how long is the word?
        new_feat[row_i, 2] = int(data[row_i, 1].isupper()) #IS IT ALL IN UPPERCASE?
        new_feat[row_i, 3] = int(data[row_i, 1].isdigit()) #it is numerical? e.g. 02-01-2002
        if row_i==0:
            new_feat[row_i, 4] = 0 # is previous word an NE?
            new_feat[row_i, 5] = '' # what's the previous POS tag?
            new_feat[row_i, 6] = '' # what's the POS tag 2 words ago?
            new_feat[row_i, 7] = 1 # is the first in the sentence?
            new_feat[row_i, 8] = 0
        else:
            new_feat[row_i, 4] = int(data[row_i-1, 3]!='O') # is previous an NE?
            new_feat[row_i, 5] = data[row_i-1, 2] # what's the previous POS tag?
            try:
                new_feat[row_i, 6] = data[row_i-2,2] # what's the POS tag 2 words ago?
            except:
                new_feat[row_i, 6] = ''
            new_feat[row_i, 7] = int(data[row_i, 0] != data[row_i - 1, 0]) # is the first in the sentence?
            new_feat[row_i, 8] = int(data[row_i-1, 2]==data[row_i, 2]) #is this word's pos tag the same as the previous one?
            
    assert len(new_feat)==len(data), "Mismatch in shape between new features length of {} and input data of len {}".format(len(new_feat),len(data))

    new_data = np.concatenate([data, new_feat], axis=1)
    
    if return_as_df:
        new_data = pd.DataFrame(new_data, columns = output_col_names)
        for col in new_data.columns:
            if type(new_data[col][0])!=str:
                new_data[col] = new_data[col].astype('int64')
    
    return new_data

def word_to_crf_features(df):
    """Takes in dataframe and produces list of lists of dicts as expected by 
    sklearn-crfsuit CRF model. 
    ---------------
    Returns list of lists of dictionaries with feature : value pairs
    Structure is as follows: 
    1. Overall list - entire data, the collection of ordered sentences;
        2. Lists - each list represents one sentence of ordered words;
            3. Dictionaries - each dictionary is the words and its features. 
    --------------------------------
    The new features will be:
    word.lower - lowercased word
    word.istitle - whether or not the word is Title Like
    len(word) - length of word string
    word.isupper - is the word all uppercased
    word.isdigit - does the word consist of digits
    word.prefix_2 - first two characters of the word
    word.suffix_2 - last two characters of the word
    word.prefix_3 - first three characters of the word
    word.suffix_3 - last three characters of the word
    word.frequency - how often does the word appear in the sentence
    word.+1_POS - POS of the next word
    word.-1_POS - POS of the previous word
    word.-2_POS - POS of the word two words before
    word.BOS - is the word the beginning of the sentence
    word.same_POS_-1 - does the current word have the same POS as the previous word
    
    """
    
    data = df.copy()
    
    # considering whether to add corpus_frequencies as well as local ones, considering the train test split problems that will arise. 
    # Perhaps should tt-split now and then 
    corpus_frequencies = FreqDist([word.lower() for word in data.iloc[:, 1]]) #creating FreqDist dict of words in all dataframe
    features_list = [] # overarching list of lists declared here
    NE_arch_list = []
    for sent_index in data['Sentence #'].unique().astype(int):
        #breaking down our data by sentence; sectioning a copy where the 
        #rows all match the current sentence, assigning it to sentence_df
        sentence_df = data.loc[data['Sentence #']==sent_index].values
        sentence_list = [] 
        NE_document_list = []
        local_frequencies= FreqDist([word.lower() for word in sentence_df[:, 1]])
        sent_length = len(sentence_df)
        for row_i in range(len(sentence_df)):
            word_feature_dict = {}
            word_feature_dict['word.lower()'] = sentence_df[row_i, 1].lower()
            word_feature_dict['word.istitle()'] = int(sentence_df[row_i, 1].istitle()) #is It Capitalised Like A Title?
            word_feature_dict['len(word)'] = len(sentence_df[row_i, 1]) # how long is the word?
            word_feature_dict['word.isupper()'] = int(sentence_df[row_i, 1].isupper()) #IS IT ALL IN UPPERCASE?
            word_feature_dict['word.isdigit()'] = int(sentence_df[row_i, 1].isdigit()) #it is numerical? e.g. 02-01-2002
            word_feature_dict['word.prefix_2'] = sentence_df[row_i, 1][:2] # what are the first two letters of the word?
            word_feature_dict['word.suffix_2'] = sentence_df[row_i, 1][-2:] # what are the last two letters of the word?
            word_feature_dict['word.prefix_3'] = sentence_df[row_i, 1][:3] # what are the first three letters of the word?
            word_feature_dict['word.suffix_3'] = sentence_df[row_i, 1][-3:] # what are the last three letters of the word?
            word_feature_dict['word.frequency'] = local_frequencies[row_i, 1]/sent_length # what is the normalised frequency of this word in sentence?
            word_feature_dict['word.frequency'] = corpus_frequencies[row_i, 1] # what is the frequency of this word in the whole dataframe?
            try:
                word_feature_dict['word.+1_POS'] = sentence_df[row_i+1,2]
            except IndexError:
                word_feature_dict['word.+1_POS'] = ''
            if row_i==0:
#                 word_feature_dict['word.-1_NE'] = 0 # is previous word an NE?
                word_feature_dict['word.-1_POS'] = '' # what's the previous POS tag?
                word_feature_dict['word.-2_POS'] = '' # what's the POS tag 2 words ago?
                word_feature_dict['word.BOS'] = 1 # is the first in the sentence?
                word_feature_dict['word.same_POS_-1'] = 0 # is this word's POS tag the same as the previous one's
            else:
#                 word_feature_dict['word.-1_NE'] = int(data[row_i-1, 3]!='O') # is previous word an NE?
                word_feature_dict['word.-1_POS'] = sentence_df[row_i-1, 2] # what's the previous POS tag?
                try:
                    word_feature_dict['word.-2_POS'] = sentence_df[row_i-2,2] # what's the POS tag 2 words ago?
                except:
                    word_feature_dict['word.-2_POS'] = '' # what's the POS tag 2 words ago?
                word_feature_dict['word.BOS'] = 0 # is the first in the sentence?
                word_feature_dict['word.same_POS_-1'] = 0 # is this word's POS tag the same as the previous one's
                
                
            sentence_list.append(word_feature_dict)
            NE_document_list.append(sentence_df[row_i, 3])
        features_list.append(sentence_list)
        NE_arch_list.append(NE_document_list)
    return features_list, NE_arch_list
        


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



def theils_u(x, y):
    """Function taken from https://github.com/shakedzy/dython
    This function takes in a set of independent var x, and 
    dependent var y and computes the Uncertainty Coefficient
    for the two. This can be thought of as *normalised mutual
    information*
    """
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x
    
    


#################################SUMMARY TABLES CREATION#####################################



#############################MODEL BUILDING, GRIDSEARCH AND PIPELINES#####################################


#############################MODEL EVALUATION (METZ, ROC CURVE, CONF_MAT)#####################################

