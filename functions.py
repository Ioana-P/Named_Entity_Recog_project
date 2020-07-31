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




#################################API-REQUESTS#####################################




#################################SCRAPING#####################################





#################################DATA TRANSFORMATION#####################################



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

