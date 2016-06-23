# coding:utf-8

import numpy as np
from sklearn.cluster import KMeans
from glove import Corpus, Glove

import simplejson as rjson
import ujson as wjson
from collections import defaultdict

import random



class GloveWrapper():

    def __init__(self, n_features=100, name="glove.model", n_epochs=30, n_threads=4, window_size=10):
        self.name = name
        self.n_features = n_features
        self.n_epochs = n_epochs
        self.n_threads = n_threads
        self.window_size = window_size
        self.is_load = False

    def fit(self, sents):
        corpus = Corpus()
        corpus.fit(sents, window=self.window_size)

        model = Glove(no_components=self.n_features, learning_rate=0.05)
        model.fit(corpus.matrix, epochs=self.n_epochs, no_threads=self.n_threads, verbose=True)
        model.add_dictionary(corpus.dictionary)
        self.model = model

        return self

    def fit_matrix(self, coo_mat):
        model = Glove(no_components=self.n_features, learning_rate=0.05)
        model.fit(coo_mat, epochs=self.n_epochs, no_threads=self.n_threads, verbose=True)
        self.model = model

        return self

    def save(self):
        self.model.save(self.name)

        return self

    def load(self):
        if not self.is_load:
            self.model = Glove.load(self.name)
            self.is_load = True

        return self

    def transform(self, docs):
        doc_narr = np.zeros((len(docs), self.n_features))+1e-9
        for i,words in enumerate(docs):
            try:
                doc_narr[i] += self.model.transform_paragraph(words, epochs=10, ignore_missing=True)
            except ValueError:
                print ",".join(docs[i])

        doc_narr = np.nan_to_num(doc_narr)

        return doc_narr

    def most_similar(self, word, number=5):
        return self.model.most_similar(word, number=number)

    def word_cluster(self, n_clusters=30):
        
        word_vectors = self.model.word_vectors
        idx2word_dict = self.model.inverse_dictionary

        km = KMeans(n_clusters=n_clusters, n_jobs=-1)
        labels = km.fit_predict(word_vectors)

        label_word_dict = defaultdict(list)

        for word_idx, label in enumerate(labels):
            label_word_dict[label].append(idx2word_dict[word_idx])

        return label_word_dict
     





