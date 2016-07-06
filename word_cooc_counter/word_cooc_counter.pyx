
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False


import numpy as np
cimport numpy as np
import scipy.sparse as sp

from libc.stdlib cimport malloc, free

from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map
# from libcpp.set cimport set as cppset
from libcpp.unordered_set cimport unordered_set as cppset
from libcpp.vector cimport vector
from libcpp.string cimport string
from cython.operator cimport dereference as deref, preincrement as inc


from pprint import pprint
import json


ctypedef unsigned long INDEX_T
ctypedef double DATA_T

cdef inline int int_min(int a, int b) nogil: return a if a <= b else b




"""
AoaMatrix START ===================================================================
"""

cdef struct SparseRowMatrix:
    vector[vector[int]] *indices
    vector[vector[DATA_T]] *data


cdef int binary_search(int* vec, int size, int first, int last, int x) nogil:
    """
    Binary seach in an array of ints
    """

    cdef int mid = 0, start_gap_end = last - first, i = 0, j = 0, col_i = 0, cur_pos = 0, insert_flag = 0
    cdef int result[2]
    result[0] = 0; result[1] = 0; 

    while 1:

        if start_gap_end <= 3:
            for i in range(first, last+1):
                col_i = vec[i]

                if x < col_i:
                    cur_pos = i
                    break

                elif x == col_i :
                    cur_pos = i
                    break

                else:
                    continue

            else:
                cur_pos = last + 1

            break

        else:

            mid = (first + last) / 2
            col_i = vec[mid]

            if x == col_i:
                cur_pos = mid
                insert_flag = 0
                break

            elif x < col_i:
                last = mid

            elif x > col_i:
                first = mid

            start_gap_end = last - first


    return cur_pos


cdef SparseRowMatrix* new_matrix():
    """
    Allocate and initialize a new matrix
    """

    cdef SparseRowMatrix* mat

    mat = <SparseRowMatrix*>malloc(sizeof(SparseRowMatrix))

    if mat == NULL:
        raise MemoryError()

    mat.indices = new vector[vector[int]]()
    mat.data = new vector[vector[DATA_T]]()

    return mat


cdef void free_matrix(SparseRowMatrix* mat) nogil:
    """
    Deallocate the data of a matrix
    """

    cdef int i
    cdef int rows = mat.indices.size()

    for i in range(rows):
        deref(mat.indices)[i].clear()
        deref(mat.data)[i].clear()

    del mat.indices
    del mat.data

    free(mat)


cdef DATA_T op_sprm_matrix(SparseRowMatrix* mat, int row, int col, DATA_T val, int incr_flag=0, int get_flag=0) nogil:
    """
    Increment the (row, col) entry of mat by increment.
    """

    cdef vector[int]* row_indices
    cdef vector[DATA_T]* row_data
    cdef int idx
    cdef int col_at_idx

    if get_flag > 0:
        if row >= mat.indices.size():
            return -2 # -2 -> no row 

    else:
        # Add new row if necessary
        while row >= mat.indices.size():
            mat.indices.push_back(vector[int]())
            mat.data.push_back(vector[DATA_T]())

    row_indices = &(deref(mat.indices)[row])
    row_data = &(deref(mat.data)[row])

    # Find the column element, or the position where
    # a new element should be inserted
    if row_indices.size() == 0:
        idx = 0
    else:
        idx = binary_search(&(deref(row_indices)[0]), row_indices.size(),
                            0, row_indices.size() - 1, col)

    if get_flag > 0:
        if idx == row_indices.size():
            return -3 # -3 -> no such col
        else:
            col_at_idx = deref(row_indices)[idx]
            if col_at_idx == col:
                return deref(row_data)[idx]
            else:
                return -3

    else:

        # Element to be added at the end
        if idx == row_indices.size():
            row_indices.insert(row_indices.begin() + idx, col)
            row_data.insert(row_data.begin() + idx, val)
            return -1

        col_at_idx = deref(row_indices)[idx]

        if col_at_idx == col:
            # Element to be incremented

            if incr_flag > 0:
                deref(row_data)[idx] = deref(row_data)[idx] + val
            else:
                deref(row_data)[idx] = val

        else:
            # Element to be inserted
            row_indices.insert(row_indices.begin() + idx, col)
            row_data.insert(row_data.begin() + idx, val)

    return -1


cdef int matrix_nnz(SparseRowMatrix* mat) nogil:
    """
    Get the number of nonzero entries in mat
    """

    cdef int i
    cdef int size = 0

    for i in range(mat.indices.size()):
        size += deref(mat.indices)[i].size()

    return size


cdef matrix_to_coo(SparseRowMatrix* mat, int shape):
    """
    Convert to a shape by shape COO matrix.
    """

    cdef int i, j
    cdef int row
    cdef int col
    cdef int rows = mat.indices.size()
    cdef int no_collocations = matrix_nnz(mat)

    # Create the constituent numpy arrays.
    row_np = np.empty(no_collocations, dtype=np.int32)
    col_np = np.empty(no_collocations, dtype=np.int32)
    data_np = np.empty(no_collocations, dtype=np.float64)
    cdef int[:] row_view = row_np
    cdef int[:] col_view = col_np
    cdef double[:] data_view = data_np

    j = 0

    for row in range(rows):
        for i in range(deref(mat.indices)[row].size()):

            row_view[j] = row
            col_view[j] = deref(mat.indices)[row][i]
            data_view[j] = deref(mat.data)[row][i]

            j += 1

    # Create and return the matrix.
    return sp.coo_matrix((data_np, (row_np, col_np)),
                         shape=(shape,
                                shape),
                         dtype=np.float64)


cdef class AoaMatrix:

    # in name of array of array
    cdef vector[vector[int]] *indices
    cdef vector[vector[DATA_T]] *data

    cdef SparseRowMatrix* sprm_p

    def __cinit__(self, n_rows=10, n_cols=10):
        self.sprm_p = new_matrix()
        self.indices = self.sprm_p.indices
        self.data = self.sprm_p.data
        
        pass

    def __dealloc__(self):
        free_matrix(self.sprm_p)

    def test_memory(self, int height, int width):
        # for testing memory usage

        self.indices = new vector[vector[int]]()
        self.data = new vector[vector[DATA_T]]()

        cdef vector[int]* row_indices
        cdef vector[DATA_T]* row_data
        
        # test memory usage
        cdef int i = 0, j = 0
        for i in range(height):
            if i >= self.indices.size():
                self.indices.push_back(vector[int]())
                self.data.push_back(vector[DATA_T]())

            row_indices = &(deref(self.indices)[i])
            row_data = &(deref(self.data)[i])

            for j in range(width):
                row_indices.push_back(j)
                row_data.push_back(10*i+j)

    def set(self, INDEX_T row_i, INDEX_T col_i, DATA_T val):
        return op_sprm_matrix(self.sprm_p, row_i, col_i, val, incr_flag=0, get_flag=0)

    def get(self, INDEX_T row_i, INDEX_T col_i):
        cdef DATA_T result = op_sprm_matrix(self.sprm_p, row_i, col_i, 0, incr_flag=0, get_flag=1)
        
        # if result == -1:
        #     return None
        # else:
        #     return result

        return result

    def incr(self, INDEX_T row_i, INDEX_T col_i, DATA_T val):
        return op_sprm_matrix(self.sprm_p, row_i, col_i, val, incr_flag=1, get_flag=0)

    def todok(self):
        cdef dict dok = {}, temp_dict = {}

        cdef int n_rows = self.indices.size(), row_i = 0, j = 0, col_i = 0, n_cols = 0
        cdef vector[int]* row_indices
        cdef vector[DATA_T]* row_data
        cdef DATA_T val = 0

        for row_i in range(n_rows):
            row_indices = &(deref(self.indices)[row_i])
            row_data = &(deref(self.data)[row_i])
            n_cols = row_indices.size()

            temp_dict = {}
            dok[row_i] = temp_dict

            for j in range(n_cols):
                col_i = deref(row_indices)[j]
                val = deref(row_data)[j]

                temp_dict[col_i] = val

        return dok

    def tocoo(self):
        cdef int max_word_id = 0

        cdef int n_rows = self.indices.size(), row_i = 0, j = 0, col_i = 0, n_cols = 0
        cdef vector[int]* row_indices
        cdef vector[DATA_T]* row_data
        cdef DATA_T val = 0

        for row_i in range(n_rows):
            row_indices = &(deref(self.indices)[row_i])
            n_cols = row_indices.size()

            if row_i > max_word_id:
                max_word_id = row_i

            for j in range(n_cols):
                col_i = deref(row_indices)[j]
                if col_i > max_word_id:
                    max_word_id = col_i

        # print "max_word_id:", max_word_id
        coo_mtx = matrix_to_coo(self.sprm_p, max_word_id+1)
        
        return coo_mtx

    def tostring(self):
        return json.dumps(self.todok(), indent=2)

    def get_aoa(self):
        cdef list aoa_indices = deref(self.indices)
        cdef list aoa_data = deref(self.data)

        return aoa_indices, aoa_data

    def __repr__(self):
        return self.tostring()

    def __str__(self):
        return self.tostring()





"""
AoaMatrix END ===================================================================
"""


cdef class WordCoocDict:

    cdef public _dict

    def __cinit__(self):
        self._dict = {}

    def add(self, str word):
        try:
            self._dict[word][1] += 1
        except KeyError:
            self._dict[word] = [len(self._dict), 1]
        # if word not in self._dict:
            
        # else:

    def shrink(self, DATA_T thres, reset_id_flag=True):
        cdef str word
        cdef list word_item
        cdef int word_id = 0, word_freq = 0, new_word_id = 0

        for word, word_item in self._dict.items():
            word_id, word_freq = word_item
            if word_freq < thres: # thres is inside boundary
                del self._dict[word]

        if reset_id_flag:
            _dict_items = sorted(self._dict.items(), key=lambda x:x[1][1], reverse=True)
            new_word_id = 0
            for word, word_item in _dict_items:
                word_item[0] = new_word_id

                new_word_id += 1

    def set(self, str word, DATA_T val):
        self._dict[word] = val
        
    def get(self, str word, placeholder=None):
        return self._dict.get(word, placeholder)

    def incr(self, str word, DATA_T val):

        try:
            self._dict[word] += val
        except KeyError:
            self._dict[word] = val




cdef class WordCoocCounter:

    cdef WordCoocDict _wcc_dict
    cdef AoaMatrix _aoa_mtx

    def __cinit__(self):
        self._wcc_dict = WordCoocDict()
        self._aoa_mtx = AoaMatrix()

        pass

    def fit_dictionary(self, words_list, min_word_freq=-1):
        cdef tuple words = ()
        cdef str word = ""
        cdef int word_id = 0
        cdef list sent_word_id_list = []

        # gen dictionary first
        for words_x in words_list:
            if type(words_x) == str:
                words = tuple(words_x.split())
            else:
                words = tuple(words_x)

            for word in words:
                self._wcc_dict.add(word)

        # shrink dictionary if necessary
        if min_word_freq > 0:
            self._wcc_dict.shrink(min_word_freq)

    def fit_corpus(self, words_list, int min_word_freq=-1, int window_size=10):

        cdef tuple words = ()
        cdef str word = ""
        cdef int word_id = 0, cur_word_id = 0, neighbor_word_id = 0, window_stop = 0
        cdef list sent_word_id_list = []
        cdef int sent_size = 0

        cdef int i = 0, j = 0

        self.fit_dictionary(words_list, min_word_freq)

        # gen cooc matrix
        for words_x in words_list:
            if type(words_x) == str:
                words = tuple(words_x.split())
            else:
                words = tuple(words_x)

            sent_word_id_list = []
            
            # words 2 ids
            for word in words:
                word_id = self._wcc_dict.get(word, [-1,0])[0]
                if word_id >= 0: # will filter some low freq word
                    sent_word_id_list.append(word_id)

            sent_size = len(sent_word_id_list)

            for i in range(sent_size):
                cur_word_id = sent_word_id_list[i]

                window_stop = int_min(i + window_size + 1, sent_size)

                for j in range(i+1, window_stop):
                    neighbor_word_id = sent_word_id_list[j]

                    if cur_word_id == neighbor_word_id: 
                        continue

                    if cur_word_id < neighbor_word_id:
                        self._aoa_mtx.incr(cur_word_id, neighbor_word_id, 1.0/(j - i))
                    else:
                        self._aoa_mtx.incr(neighbor_word_id, cur_word_id, 1.0/(j - i))

        return self

    def get_dictionary(self):
        return self._wcc_dict._dict

    def get_aoa_mtx(self):
        return self._aoa_mtx

    def tocoo(self):
        return self._aoa_mtx.tocoo()

        




