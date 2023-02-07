import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

raw = open('rand_input.txt').read()
raw = nltk.Text(nltk.word_tokenize(raw))
fdist = nltk.FreqDist(raw)
print (fdist.N())

vectorizer = CountVectorizer(min_df=1)
