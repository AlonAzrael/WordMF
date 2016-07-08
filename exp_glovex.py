


import numpy as np
from numpy import random
from scipy.sparse import random as random_mtx, coo_matrix
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from word_cooc_counter import WordCoocCounter

from glove import *

from time import time
import os
import cPickle as pickle
from collections import defaultdict



N_FEATURES = 200



def load_coo_mtx():
    with open("./__X_data/word_coo_mtx.pkl", "rb") as F:
        coo_mtx = pickle.load(F)

    print coo_mtx
    print coo_mtx.shape

    return coo_mtx


def load_id2words():
    with open("./__X_data/wcc_dict", "r") as F:
        wcc_dict_s = F.read()
        id2words_mapping = eval(wcc_dict_s)

    return id2words_mapping


def train_model(coo_mtx):
    start_time = time()

    # model = GloveWrapper(n_features=100, n_threads=4, n_epochs=30)
    # model.fit_matrix(coo_mtx)
    model = Glove(no_components=N_FEATURES, n_threads=4, n_epochs=30, )
    model.fit(coo_mtx, shrink_symm=False, iter_counter=1, k_loss=2, log_flag=True)

    end_time = time()
    print "elapsed time:", end_time - start_time

    # print model.word_vectors
    word_vectors = model.word_vectors
    print word_vectors.shape

    return model


def save_model(model):
    model.save("./__X_model/glovex_model.pkl")

def load_model():
    return Glove.load("./__X_model/glovex_model.pkl")


def save_tsne_coor(word_tsne_coor=None, id2words_mapping=None):
    if word_tsne_coor is not None:
        s_list = []
        with open("./__X_data/word_tsne_coor.txt", "w") as F :
            for word_id, tsne_coor in enumerate(word_tsne_coor):
                word, word_freq = id2words_mapping[word_id]
                s_list.append( word +" "+str(word_freq)+" "+str(tsne_coor[0])+","+str(tsne_coor[1]) )

            F.write("\n".join(s_list))


def gen_tsne_coor(word_vectors):
    tsne = TSNE(n_components=2)
    word_tsne_coor = tsne.fit_transform(word_vectors)

    return word_tsne_coor


def cluster_words(word_vectors, words, n_clusters=5):
    km = KMeans(n_clusters=n_clusters, n_jobs=-1)
    labels = km.fit_predict(word_vectors)

    label_word_dict = defaultdict(list)
    
    for i, label in enumerate(labels):
        label_word_dict[label].append(words[i])

    return label_word_dict



if __name__ == '__main__':
    # coo_mtx = load_coo_mtx()
    id2words_mapping = load_id2words()

    # model = train_model(coo_mtx)
    # save_model(model)

    # 
    model = load_model()
    with open("./__X_data/keyword_ids", "r") as F:
        keyword_ids = eval(F.read())

    word_vectors = model.word_vectors[keyword_ids]
    print word_vectors
    # word_tsne_coor = gen_tsne_coor(word_vectors)
    # save_tsne_coor( word_tsne_coor, [id2words_mapping[ki] for ki in keyword_ids] )

    words = [id2words_mapping[ki][0] for ki in keyword_ids]
    for label, words in cluster_words(word_vectors, words).iteritems():
        print label, ":", " ".join(words)





