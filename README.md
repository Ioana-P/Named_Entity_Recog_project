# Named Entity Recognition classifier in pytorch and sklearn

#### Given a set of documents, can we identify what the most important entities mentioned are? 

##### Can we classify terms as relating to dates, times, organisations and geopolitical entities so that we can take any text (such as a news article) and automatically extract information and use it to populate summary tables or to help search queries? 
Moreover, how successfully can we train a model to identify the entirety of an NE token and not over-tagging or under-tagging words around the NE?

##### The most successful developments in this area have employed neural network architectures, which is what I will be using in this project[1]. 

###### In this project I will have built a baseline feature-based model to serve as a comparison and for interpreting feature importance and detecting underlying statistical trends in our data. The majority of the project will be focused on building a Recurrent Neural Network (RNN) in pytorch and improving that model through hyperparameter optimisation and adding an attention layer. 
##### By the end I hope to have achieved a reasonable neural classifier (with an macro-averaged F1 score above 0.8) that utilises an attention layer to capture inter-word dependencies. As an additional goal, I would like to deploy this model to via AWS to a web app that can interactively highlight NEs within the text. 
_____________________________________________________________________________________________________________________________


### Additional Background Notes
A named entity (NE) is a word or group of words “that clearly identifies one item from a set of other items that have similar attributes”[1]. Identifying and labelling important NEs in text data is a key challenge in Natural Language Processing (NLP). NER has many useful applications in the wider NLP space, such as creating new data features for Information Extraction systems, improving performance in Machine Translation and Summarisation systems or simply providing a way to highlight relevant entities within a text for a reader’s convenience[1]. Solutions to the problem can be categorised in four groups: rule-based algorithms, unsupervised learning approaches, feature-based supervised learners and deep learning techniques. This project will focus on the last two.

(LINK to youtube video; and/or blog post)

###  Significant findings/ Executive Summary

## OSEMN

### OBTAIN
- download dataset

### SCRUB
- dataset is very clean, as has been confirmed by multiple previous users on Kaggle

### EXPLORE
- visualise target category distributions; use spacy’s EDA methods to show NER in action, as shown in notebook 2; 

### MODEL
- after performing a train-validation-test split, I transform the data to be adequate for a feature-based CRF model. This will involve turning text data into an array of numbers encoding certain features (e.g. {text.isupper(), text.istitle(), len(text), … }, interactions of those features and the provided POS tag for the word. Use GridSearchCV to cross-validate and optimise. 
 - Use a pytorch RNN-based model to perform NER. I will build the preprocessing and vectorisation steps required for adequately preparing the data for this task. The first neural network layer will be the Lookup layer that matches the token to the pre-trained GloVe Vectors [3]. Using pre-trained language model vectors like GloVe has been shown consistently to improve model performance across a variety of NLP tasks.  Time-permitting I will add an Attention Layer to the neural model. The training for this model will occur in AWS Sagemaker.
- To help with correctly measuring performance, I’ll be using the **seqeval** library for my measures of F1 and precision.

### INTERPRET - 
- CRF model will be easier to interpret and I’ll be able to extract interesting dependencies between created features and the target values. The aforementioned Kaggle post shows one method of doing this. Knowing such dependencies will also be useful in gauging how successful the model will be on future, unseen data. 
- To interpret the neural model, I will pass in more recent news articles into it to check which entities in particular are (mis)classified as which categories. If it were possible, it would be great to find and use a python-friendly text highlighter that would encode predictions directly in the text. 



### The Data at a Glance (EDA)

_____________________________________________________________________________________________________________________________


### Model performance on validation and test data


_____________________________________________________________________________________________________________________________

### Model interpretation


_____________________________________________________________________________________________________________________________


### Insights and possible actions

### Filing system:

Notebooks
1. Modelling_and_insights.ipynb - principal notebook of findings and final results; most relevant notebook to most people
2. Model_building_and_optimization.ipynb - different models were all tried out here, tuned and optimized
3. EDA.ipynb - exploration of the data; go here for further interesting visuals
4. data_cleaning.ipynb - notebook detailing the entire cleaning process

Folders
* clean_data - folder containing clean data and any additional summary data generated; feature-based data and vectorised representations will be stored here
* raw_data - data as it was when imported / downloaded / scraped into my repository
* archive - any additional files and subfolders will be here
* glove - folder containing Global Word Vector Representations (as produced by the Stanford NLP group)[2]


References:
1. @misc{li2018survey,
    title={A Survey on Deep Learning for Named Entity Recognition},
    author={Jing Li and Aixin Sun and Jianglei Han and Chenliang Li},
    year={2018},
    eprint={1812.09449},
    archivePrefix={arXiv},
    primaryClass={cs.CL}}
2. Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. [pdf] [bib]
3. 
4. 



