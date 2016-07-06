


import numpy as np
from numpy import random
from scipy.sparse import random as random_mtx, coo_matrix

from word_cooc_counter import WordCoocCounter

from glove import *

from time import time
import os
import cPickle as pickle



N_FEATURES = 100



def load_coo_mtx():
    with open("./word_coo_mtx.pkl", "rb") as F:
        coo_mtx = pickle.load(F)

    print coo_mtx
    print coo_mtx.shape

    return coo_mtx

def train_glove(coo_mtx):
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



if __name__ == '__main__':
    coo_mtx = load_coo_mtx()
    train_glove(coo_mtx)


