


import numpy as np
from numpy import random
from scipy.sparse import random as random_mtx, coo_matrix
from sklearn.manifold import TSNE

from word_cooc_counter import WordCoocCounter

from glove import *

from time import time
import os
import cPickle as pickle



N_FEATURES = 100



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
    model = Glove(no_components=N_FEATURES, n_threads=4, n_epochs=20, )
    model.fit(coo_mtx, shrink_symm=True, iter_counter=1, k_loss=2, log_flag=True)

    end_time = time()
    print "elapsed time:", end_time - start_time

    # print model.word_vectors
    word_vectors = model.word_vectors
    print word_vectors.shape

    return model


def save_all(model, word_tsne_coor=None, id2words_mapping=None):
    model.save("./__X_model/glovex_model.pkl")

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




if __name__ == '__main__':
    coo_mtx = load_coo_mtx()
    id2words_mapping = load_id2words()

    model = train_model(coo_mtx)
    word_tsne_coor = gen_tsne_coor(model.word_vectors)
    save_all(model, word_tsne_coor, id2words_mapping)







