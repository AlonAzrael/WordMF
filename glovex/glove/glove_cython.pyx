#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as np
import scipy.sparse as sp
import collections
from cython.parallel cimport parallel, prange


cdef inline double double_min(double a, double b) nogil: return a if a <= b else b
cdef inline int int_min(int a, int b) nogil: return a if a <= b else b
cdef inline int int_max(int a, int b) nogil: return a if a > b else b


cdef extern from "math.h" nogil:
    double sqrt(double)
    double c_log "log"(double)


def fit_vectors(double[:, ::1] wordvec,
                double[:, ::1] wordvec_sum_gradients,
                double[::1] wordbias,
                double[::1] wordbias_sum_gradients,
                int[::1] row,
                int[::1] col,
                double[::1] counts,
                int[::1] shuffle_indices,
                double initial_learning_rate,
                double max_count,
                double alpha,
                double max_loss,
                int no_threads, 
                int iter_counter=1, 
                int k_loss=2):
    """
    Estimate GloVe word embeddings given the cooccurrence matrix.
    Modifies the word vector and word bias array in-place.

    Training is performed via asynchronous stochastic gradient descent,
    using the AdaGrad per-coordinate learning rate.
    """

    # print "fit_vectors"

    # Get number of latent dimensions and
    # number of cooccurrences.
    cdef int dim = wordvec.shape[1]
    cdef int no_cooccurrences = row.shape[0]

    # Hold indices of current words and
    # the cooccurrence count.
    cdef int word_a, word_b
    cdef double count, learning_rate, gradient

    # Loss and gradient variables.
    cdef double prediction, entry_weight, loss
    # max_count = c_log(max_count)
    # max_count = c_log(100)

    # Iteration variables
    cdef int i, j, shuffle_index
    cdef int two_iter_i = 0

    # We iterate over random indices to simulate
    # shuffling the cooccurrence matrix.
    # print no_cooccurrences
    with nogil:
        for j in prange(no_cooccurrences, num_threads=no_threads,
                        schedule='static'):
            shuffle_index = shuffle_indices[j]
            count = counts[shuffle_index]
            word_a = row[shuffle_index]
            word_b = col[shuffle_index]

            # do double time, and it seems like not only faster but lower error_sq
            for two_iter_i in range(iter_counter):

                if two_iter_i % 2 == 0:
                    word_a = row[shuffle_index]
                    word_b = col[shuffle_index]
                else:
                    word_b = row[shuffle_index]
                    word_a = col[shuffle_index]

                # Get prediction
                prediction = 0.0

                for i in range(dim):
                    prediction = prediction + wordvec[word_a, i] * wordvec[word_b, i]

                prediction = prediction + wordbias[word_a] + wordbias[word_b]

                # Compute loss and the example weight.
                # log version
                # entry_weight = double_min(1.0, (count / max_count)) ** alpha
                # loss = entry_weight * (prediction - c_log(count)) 
                # no log version
                # entry_weight = double_min(1.0, (count / max_count)) 

                entry_weight = count / max_count
                if entry_weight > 2:
                    entry_weight = 2
                elif entry_weight < 0.01:
                    continue

                # entry_weight = 1 # the matrix should be filtered, so all words are important
                loss = entry_weight * (prediction - count) * k_loss

                # Clip the loss for numerical stability.
                if loss < -max_loss:
                    loss = -max_loss
                elif loss > max_loss:
                    loss = max_loss

                # Update step: apply gradients and reproject
                # onto the unit sphere.
                for i in range(dim):

                    learning_rate = initial_learning_rate / sqrt(wordvec_sum_gradients[word_a, i])
                    gradient = loss * wordvec[word_b, i]
                    wordvec[word_a, i] = (wordvec[word_a, i] - learning_rate 
                                          * gradient)
                    wordvec_sum_gradients[word_a, i] += gradient ** 2

                    learning_rate = initial_learning_rate / sqrt(wordvec_sum_gradients[word_b, i])
                    gradient = loss * wordvec[word_a, i]
                    wordvec[word_b, i] = (wordvec[word_b, i] - learning_rate
                                          * gradient)
                    wordvec_sum_gradients[word_b, i] += gradient ** 2

                # Update word biases.
                learning_rate = initial_learning_rate / sqrt(wordbias_sum_gradients[word_a])
                wordbias[word_a] -= learning_rate * loss
                wordbias_sum_gradients[word_a] += loss ** 2

                learning_rate = initial_learning_rate / sqrt(wordbias_sum_gradients[word_b])
                wordbias[word_b] -= learning_rate * loss
                wordbias_sum_gradients[word_b] += loss ** 2


def transform_paragraph(double[:, ::1] wordvec,
                        double[::1] wordbias,
                        double[::1] paragraphvec,
                        double[::1] sum_gradients,
                        int[::1] row,
                        double[::1] counts,
                        int[::1] shuffle_indices,
                        double initial_learning_rate,
                        double max_count,
                        double alpha,
                        int epochs):
    """
    Compute a vector representation of a paragraph. This has
    the effect of making the paragraph vector close to words
    that occur in it. The representation should be more
    similar to words that occur in it multiple times, and
    less close to words that are common in the corpus (have
    large word bias values).

    This should be be similar to a tf-idf weighting.
    """

    # Get number of latent dimensions and
    # number of cooccurrences.
    cdef int dim = wordvec.shape[1]
    cdef int no_cooccurrences = row.shape[0]

    # Hold indices of current words and
    # the cooccurrence count.
    cdef int word_b, word_a
    cdef double count

    # Loss and gradient variables.
    cdef double prediction
    cdef double entry_weight
    cdef double loss
    cdef double gradient

    # Iteration variables
    cdef int epoch, i, j, shuffle_index

    # We iterate over random indices to simulate
    # shuffling the cooccurrence matrix.
    for epoch in range(epochs):
        for j in range(no_cooccurrences):
            shuffle_index = shuffle_indices[j]

            word_b = row[shuffle_index]
            count = counts[shuffle_index]

            # Get prediction
            prediction = 0.0
            for i in range(dim):
                prediction = prediction + paragraphvec[i] * wordvec[word_b, i]
            prediction += wordbias[word_b]

            # Compute loss and the example weight.
            entry_weight = double_min(1.0, (count / max_count)) ** alpha
            loss = entry_weight * (prediction - c_log(count))

            # Update step: apply gradients.
            for i in range(dim):
                learning_rate = initial_learning_rate / sqrt(sum_gradients[i])
                gradient = loss * wordvec[word_b, i]
                paragraphvec[i] = (paragraphvec[i] - learning_rate
                                   * gradient)
                sum_gradients[i] += gradient ** 2




# little function for benchmarking numpy and c-array


def mandel_cython_carray(int n=400,int maxi=512):
    
    cdef double x0 = -2.0
    cdef double x1 = 1.0
    cdef double y0 = -1.0
    cdef double y1 = 1.0

    # declare the type and dimension of numpy arrays
    # (and create them in the same line, C-style)
    cdef double[::1] xs = np.linspace(x0,x1,n)
    cdef double[::1] ys = np.linspace(y0,y1,n)

    cdef double[:, ::1] escape = np.ones((n,n),'float64') + 2
    cdef np.ndarray[double,ndim=1] escape_np

    # declare integer counters
    cdef int i,j,it,esc, _

    # declare complex variables
    cdef double complex z,c

    # use classic c-loop 
    for _ in range(maxi):
        for i in range(n):
            for j in range(n):
                escape[i,j] += xs[j]*ys[j]

    if 0:
        for i in range(n):
            for j in range(n):
                z = 0 + 0j
                c = xs[i] + 1j * ys[j]

                esc = maxi
                for it in range(maxi):
                    z = z*z + c

                    # let's allow ourselves one hand-tuned optimization,
                    # which avoids the sqrt implicit in abs
                    if z.real*z.real + z.imag*z.imag > 4:
                        esc = it
                        break

                escape[j,i] = esc

    return np.asarray(escape)


def mandel_cython_np(int n=400,int maxi=512):
    
    cdef double x0 = -2.0, x1 = 1.0, y0 = -1.0, y1 = 1.0

    # declare the type and dimension of numpy arrays
    # (and create them in the same line, C-style)
    cdef np.ndarray[double,ndim=1] xs = np.linspace(x0,x1,n)
    cdef np.ndarray[double,ndim=1] ys = np.linspace(y0,y1,n)
    cdef np.ndarray[double,ndim=1] temps

    cdef np.ndarray[double,ndim=2] escape = np.ones((n,n),'float64') + 2

    # declare integer counters
    cdef int i,j,it,esc

    # declare complex variables
    cdef double complex z,c

    # use np vec-op
    temps = (xs*ys)*maxi
    escape += temps

    if 0:
        for i in range(n):
            for j in range(n):
                z = 0 + 0j
                c = xs[i] + 1j * ys[j]

                esc = maxi
                for it in range(maxi):
                    z = z*z + c

                    # let's allow ourselves one hand-tuned optimization,
                    # which avoids the sqrt implicit in abs
                    if z.real*z.real + z.imag*z.imag > 4:
                        esc = it
                        break

                escape[j,i] = esc

    return escape


