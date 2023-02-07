# -*- coding: utf-8 -*-
"""
Basic word2vec using Gensim

"""

import gensim.downloader as api
wv = api.load('word2vec-google-news-300')

#TODO: Find out how to save and load word2vec model, so that you don't need to download is again and again

"""Retrieve the vocabulary of a model"""

for index, word in enumerate(wv.index_to_key):
    if index == 10:
        break
    print(f"word #{index}/{len(wv.index_to_key)} is {word}")

"""Get Vector representation of a word

word = 'king'
vec_king = wv[word]

Calculate Word Similarity
"""

pairs = [
    ('car', 'minivan'),   # a minivan is a kind of car
    ('car', 'bicycle'),   # still a wheeled vehicle
    ('car', 'airplane'),  # ok, no wheels, but still a vehicle
    ('car', 'cereal'),    # ... and so on
    ('car', 'communism'),
]
for w1, w2 in pairs:
    print('%r\t%r\t%.2f' % (w1, w2, wv.similarity(w1, w2)))

"""Nearest words to give words"""

words = ['car', 'minivan']
print(wv.most_similar(positive= word, topn=5))