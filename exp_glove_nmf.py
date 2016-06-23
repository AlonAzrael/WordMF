


import numpy as np
from numpy import random
from scipy.sparse import random as random_mtx, coo_matrix

from glovezh import *
from glove import *
from sklearn.decomposition import NMF
from sklearn.datasets import make_blobs

from time import time
import os

N_WORD = 10000
N_FEATURES = 100


def gen_random_wh(n_word=N_WORD, n_features=N_FEATURES):
    return make_blobs(n_samples=n_word, n_features=n_features, centers=10, shuffle=False, cluster_std=10.0)[0]


def random_mtx_sqaure(height, width, density=0.1, format="coo"):
    assert height == width

    # W = gen_random_wh(height)
    
    row = []
    col = []
    data = []

    total_size = height*width
    nnz = int(total_size*density)
    if nnz <= height:
        nnz = height+1

    for i in xrange(height):
        row_i = i
        col_i = random.randint(height)
        x = random.randint(1, 180)
        # x = W[row_i]*W[]

        row.append(row_i)
        col.append(col_i)
        data.append(x)

        row.append(col_i)
        col.append(row_i)
        data.append(x)

    for i in xrange(height, nnz):
        row_i = random.randint(height)
        col_i = random.randint(height)
        x = random.randint(1, 180)
        # x = random.random()

        row.append(row_i)
        col.append(col_i)
        data.append(x)

        row.append(col_i)
        col.append(row_i)
        data.append(x)

    row = np.asarray(row)
    col = np.asarray(col)
    data = np.asarray(data)

    coo_mtx = coo_matrix((data, (row, col)), shape=(height, width))

    return coo_mtx


def gen_random_coo_mtx(size=100, density=0.01, filebase=None):
    # coo_mtx = random_mtx(size, size, density=0.2, format="coo", )
    coo_mtx = random_mtx_sqaure(size, size, density=density, format="coo")
    # coo_mtx *= 100
    # coo_mtx.data = coo_mtx.data + random.randint(100, size=len(coo_mtx.data))
    # coo_mtx += 80

    # coo_mtx_log = coo_mtx.copy()
    # coo_mtx_log.data = np.log(coo_mtx_log.data)

    coo_mtx_log = coo_mtx.log1p()

    mtx_str_list = []
    for mtx in [coo_mtx, coo_mtx_log]:
        lines = zip(mtx.row+1, mtx.col+1, mtx.data)
        mtx_str_list.append("\n".join([" ".join([str(row), str(col), str(data)]) for row,col,data in lines]))

    if filebase is not None:
        
        for filename, mtx_str in zip(["coo_rand", "coo_rand_log"], mtx_str_list):
            fliepath = filebase+"/"+filename+".mtx"
            filedir = os.path.dirname(filepath)
            
            if not os.path.exists(filedir):
                os.makedirs(filedir)
            
            with open(filepath, "w") as F:
                F.write("%%MatrixMarket matrix coordinate real general\n")
                F.write( " ".join([str(x) for x in [coo_mtx.shape[0], coo_mtx.shape[1], len(coo_mtx.data)]])+"\n" ) 
                F.write(mtx_str)


def load_coo_mtx(filepath, upper_tri=False):
    # mtx format index is start from 1
    with open(filepath, "r") as F:
        F.readline()
        format = [int(x) for x in F.readline().split()]
        row = []
        col = []
        data = []
        for line in F.xreadlines():
            r,c,d = line.split()
            row.append(int(r))
            col.append(int(c))
            data.append(np.float(d))

        row = np.asarray(row) - 1
        col = np.asarray(col) - 1
        data = np.asarray(data)

        # print type(row)
        # print type(row[::2])

        if upper_tri:
            row = row[::2].copy()
            col = col[::2].copy()
            data = data[::2].copy()

        coo_mtx = coo_matrix((data, (row, col)), shape=format[:2])

    return coo_mtx


def load_smallk_wh(wfp, hfp):
    w = np.genfromtxt(wfp, delimiter=',')
    h = np.genfromtxt(hfp, delimiter=',').T

    return w, h


def test_coo_mtx():
    row  = np.array([0, 3, 1, 0])
    col  = np.array([0, 3, 1, 2])
    data = np.array([4, 7, 7, 9])
    coo_mtx = coo_matrix((data, (row, col)), shape=(4, 4))

    print coo_mtx.data
    print coo_mtx.todense()


def score_accuracy(coo_mtx, w, biases=None, log_flag=True):

    if log_flag:
        coo_mtx_log = coo_mtx.log1p()
    else:
        coo_mtx_log = coo_mtx

    if biases is None:
        biases = np.zeros(len(w[0]))

    errors = []
    errors_sq = 0
    for i,row_i in enumerate(coo_mtx_log.row):
        col_i = coo_mtx_log.col[i]
        # error = np.dot(w[row_i],w[col_i]) + biases[row_i] + biases[col_i] - coo_mtx_log.data[i]
        error = np.dot(w[row_i],w[col_i]) - coo_mtx_log.data[i]
        errors_sq += error**2

    return errors_sq


def score_accuracy_smallk(coo_mtx, w, h, log_flag=True):
    if log_flag:
        coo_mtx_log = coo_mtx.log1p()
    else:
        coo_mtx_log = coo_mtx

    errors_sq = 0
    for i,row_i in enumerate(coo_mtx_log.row):
        col_i = coo_mtx_log.col[i]
        error = np.dot(w[row_i],h[col_i]) - coo_mtx_log.data[i]
        errors_sq += error**2

    return errors_sq


def benchmark_glovewrapper(coo_mtx):
    start_time = time()

    # model = GloveWrapper(n_features=100, n_threads=4, n_epochs=30)
    # model.fit_matrix(coo_mtx)
    model = Glove(no_components=N_FEATURES, n_threads=4, n_epochs=20, )
    model.fit(coo_mtx, shrink_symm=True)

    end_time = time()
    print "elapsed time:", end_time - start_time

    # print model.word_vectors
    word_vectors = model.word_vectors
    print word_vectors.shape
    np.savetxt("./dir_random_coo_mtx/wgl.csv", word_vectors, delimiter=",", fmt='%.9e')
    np.savetxt("./dir_random_coo_mtx/hgl.csv", word_vectors.T, delimiter=",", fmt='%.9e')

    print "score_accuracy:", score_accuracy(coo_mtx, word_vectors, model.word_biases, log_flag=False)
    # print "score_accuracy:", score_accuracy_smallk(coo_mtx, model.model.word_vectors, model.model.word_vectors, log_flag=False)


def benchmark_sklearn_nmf(coo_mtx):
    start_time = time()

    nmf = NMF(n_components=N_FEATURES, max_iter=60, shuffle=True, )
    w = nmf.fit_transform(coo_mtx)
    h = nmf.components_.T
    # print w.shape, h.shape

    end_time = time()
    print "elapsed time:", end_time - start_time

    print "score_accuracy:", score_accuracy_smallk(coo_mtx, w, h, log_flag=False)
    # print "score_accuracy:", nmf.reconstruction_err_


def benchmark_smallk_nmf(coo_mtx):
    cmd = "nmf --matrixfile ./dir_random_coo_mtx/coo_rand_log.mtx --k 100 --maxthreads 4 --algorithm MU --outfile_W ./dir_random_coo_mtx/w.csv --outfile_H ./dir_random_coo_mtx/h.csv --normalize 1 --maxiter 50"
    cmd += " --infile_W ./dir_random_coo_mtx/wgl.csv --infile_H ./dir_random_coo_mtx/hgl.csv"
    os.system(cmd)

    w,h = load_smallk_wh("./dir_random_coo_mtx/w.csv", "./dir_random_coo_mtx/h.csv")

    print "score_accuracy:", score_accuracy_smallk(coo_mtx, w, h, log_flag=False)


def benchmark_np_and_carray():
    start_time = time()
    
    # print mandel_cython_carray(n=1000, maxi=512*4)
    print mandel_cython_np(n=1000, maxi=512*4)

    end_time = time()
    print "elapsed time:", end_time - start_time




if __name__ == '__main__':
    # gen_random_coo_mtx(size=10000, density=0.01, filebase="./dir_random_coo_mtx")
    coo_mtx = load_coo_mtx(filepath="./dir_random_coo_mtx/coo_rand_log.mtx", )
    benchmark_glovewrapper(coo_mtx)
    # benchmark_smallk_nmf(coo_mtx)
    # benchmark_sklearn_nmf(coo_mtx)
    
    # load_smallk_wh("./dir_random_coo_mtx/w.csv", "./dir_random_coo_mtx/h.csv")
    # benchmark_np_and_carray()

