# -*- coding: utf-8 -*-
"""
Sample code for Topic Modelling using Gensim

"""

# This is needed if spacy's model is not installed
!python -m spacy download en_core_web_lg

import spacy
spacy.load('en_core_web_lg')
from spacy.lang.en import English
parser = English()

#Basic tokenizer
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn

#Lemmatization
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

#Combining tokenizer, lemmatization and stopword together
def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

import sklearn
# Use 20newsgropups data
groups = [
    'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']
train_data = sklearn.datasets.fetch_20newsgroups(subset="train",
                                                 categories=groups)
print("Number of training posts in tech groups:", len(train_data.filenames))

import re
text_data = []

data = train_data.data

# Some more pre-processing
#TODO:  Do this to `prepare_text_for_lda'

# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

for tmp_data in data:
  tokens = prepare_text_for_lda(tmp_data)
  text_data.append(tokens)

from gensim import corpora
#Let's create data to the gensim format
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]

#Find out how we could save this and load, so that we don't have to do same multiple time

import gensim
# Now let's do Topic Modelling
NUM_TOPICS = 25
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)

# Testing Topic Modelling
new_doc = 'Practical Bayesian Optimization of Machine Learning Algorithms'
new_doc = prepare_text_for_lda(new_doc)
new_doc_bow = dictionary.doc2bow(new_doc)
print(new_doc_bow)
print(ldamodel.get_document_topics(new_doc_bow))

# Commented out IPython magic to ensure Python compatibility.
# Install pyLDAvis for visualization

# !pip install pyLDAvis
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis  # don't skip this

import matplotlib.pyplot as plt
# %matplotlib inline

pyLDAvis.enable_notebook()
vis = gensimvis.prepare(ldamodel, corpus, dictionary)
vis

#TODO: Try different numbers of topics and observe the changes
#TODO: add bi-gram and tri-gram as text representation
#TODO: Make code more modular 