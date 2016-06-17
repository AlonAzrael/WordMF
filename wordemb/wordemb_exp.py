# coding: utf-8


from gensim.models import Word2Vec
import numpy as np
from sklearn.cluster import KMeans
from glove import Corpus, Glove

import simplejson as rjson
import ujson as wjson
from collections import defaultdict

import random


def load_corpus():
    with open("../Reduced/sents.txt") as F:
        sents = [line.split() for line in F.readlines()]

    return sents


class Word2VecWrapper():

    def __init__(self, n_features=100, name="word2vec.model"):
        self.name = name
        self.n_features = n_features

    def fit(self, sents):
        model = Word2Vec(sents, size=self.n_features, window=10, min_count=3, workers=4)
        self.model = model
        self.index2word_set = set(self.model.index2word)

        return self

    def save(self):
        self.model.save(self.name)

        return self

    def load(self):
        self.model = Word2Vec.load(self.name)
        self.index2word_set = set(self.model.index2word)

        return self

    def transform(self, docs):
        doc_narr = np.zeros((len(docs), self.n_features))+1e-9
        for i,words in enumerate(docs):
            
            vec = doc_narr[i]
            n_words = 0
            
            for word in words:
                if word in self.index2word_set:
                    vec += self.model[word]
                    n_words += 1

            # no dividing zero
            if n_words == 0:
                n_words = 1

            doc_narr[i] = np.divide(vec, n_words)

        return doc_narr

    def most_similar(self, *args, **kwargs):
        return self.most_similar(*args, **kwargs)


class GloveWrapper():

    def __init__(self, n_features=100, name="glove.model", n_epochs=30, n_threads=4, window_size=10):
        self.name = name
        self.n_features = n_features
        self.n_epochs = n_epochs
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
        


        


def word2vec_train():
    sents = load_corpus()
    Word2VecWrapper().fit(sents).save()

def glove2vec_train():
    sents = load_corpus()
    GloveWrapper().fit(sents).save()

def train_modelwrapper(modelwrapper):
    sents = load_corpus()
    modelwrapper.fit(sents).save()


def clustering_sents(modelwrapper):
    sents = load_corpus()
    doc_narr = modelwrapper.load().transform(sents)

    km = KMeans(n_clusters=30, n_jobs=-1)
    labels = km.fit_predict(doc_narr)

    sent_label_dict = defaultdict(list)
    for i,label in enumerate(labels):
        sent_label_dict[label].append(sents[i])

    # with open("./sent_label_dict.json.txt", "w") as F:
    #     wjson.dump(sent_label_dict, F, ensure_ascii=False, )

    with open("./sent_label_dict.json.txt", "w") as F:
        for label, sents in sent_label_dict.iteritems():
            F.write(str(label)+"\n")
            random.shuffle(sents)
            for sent in sents[:5]:
                F.write(",".join(sent)+"\n")


def test_most_similar(modelwrapper):
    modelwrapper.load()
    words = "军事 政治 音乐 电影 社会 足球 篮球 华为".split()
    
    for keyword in words:
        print "topic keyword:", keyword
        try:
            for word, weight in modelwrapper.most_similar(keyword, number=10):
                print word, weight
            print "----------"
        except Exception as e: 
            print "no such word", keyword


def test_word_cluster(modelwrapper):
    label_word_dict = modelwrapper.word_cluster()
    with open("./word_cluster.txt", "w") as F:
        for label, words in label_word_dict.iteritems():
            F.write("label:"+str(label)+"\n")
            F.write(",".join(words)+"\n")
            F.write("===========================\n")





if __name__ == '__main__':
    modelwrapper = GloveWrapper().load()

    # train_modelwrapper(modelwrapper)

    # clustering_sents(modelwrapper)
    # test_most_similar(modelwrapper)
    test_word_cluster(modelwrapper)

