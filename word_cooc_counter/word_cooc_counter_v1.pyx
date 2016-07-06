
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
from cython.operator cimport dereference as deref, preincrement as inc


from pprint import pprint
import json


ctypedef unsigned long INDEX_T
ctypedef double DATA_T





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
    data_np = np.empty(no_collocations, dtype=np.DATA_T64)
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
                         dtype=np.DATA_T64)


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

        # coo_mtx = matrix_to_coo(self.sprm_p, self.indices.size())
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






cdef inline int _calc_n_digits(unsigned long hash_val):
    
    cdef int n = 1
    # cdef unsigned long hash_val = <unsigned long>hash_val_l

    if hash_val >= 10000000000000000 :
        hash_val = hash_val / 10000000000000000
        n += 16
    if hash_val >= 100000000:
        hash_val /= 100000000
        n += 8
    if hash_val >= 10000:
        hash_val /= 10000
        n += 4
    if hash_val >= 100:
        hash_val /= 100
        n += 2
    if hash_val >= 10:
        hash_val /= 10
        n += 1

    return n



# cdef struct IndicesBucket:

#     vector[INDEX_T]* rowcol_buckets[21][10]
#     vector[DATA_T]* data_buckets[21][10]
    # int bucket_size[21][10]


cdef class SuzuMatrix:

    # cdef map[unsigned long, double] dok_dict
    # cdef map[int, DATA_T] dok_dict_ab
    # cdef map[int, DATA_T] dok_dict_ba

    # cdef unsigned long* indices
    # cdef double* datas

    # cdef vector[vector[INDEX_T*]]* indices_bucket_list
    cdef vector[INDEX_T]* rowcol_buckets[21][10]
    cdef vector[DATA_T]* data_buckets[21][10]

    def __cinit__(self, n_rows=10, n_cols=10):
        # self.indices = <long*>malloc(n_rows*n_cols*sizeof(long))
        # self.datas = <double*>malloc(n_rows*n_cols*sizeof(double))

        # cdef int i = 0
        # for i in range(n_rows*n_cols):
        #     self.indices[i] = i
        #     self.datas[i] = 1.0

        cdef vector[INDEX_T]* temp_rowcol_vector 
        cdef vector[DATA_T]* temp_data_vector 

        cdef int i = 0, j = 0
        for i in range(1, 21):
            for j in range(1, 10):
                self.rowcol_buckets[i][j] = new vector[INDEX_T]()
                self.data_buckets[i][j] = new vector[DATA_T]()

        # print "init status"
        # pprint(self.status())

        pass

    def __dealloc__(self):
        cdef int i = 0, j = 0
        for i in range(1, 21):
            for j in range(1, 10):
                del self.rowcol_buckets[i][j]
                del self.data_buckets[i][j]

    def calc_n_digits(self, unsigned long number):
        cdef int n_digits = 0
        n_digits = _calc_n_digits(number)

        return n_digits

    def _bisect_op(self, INDEX_T row_i, INDEX_T col_i, DATA_T val=1, incr_flag=0, get_flag=0):
        cdef INDEX_T hash_val = 0

        if 0:
            if row_i >= col_i:
                hash_val = row_i * row_i + row_i + col_i
                # self.dok_dict_ab[hash_val] = val
            else: # mostly use this
                hash_val = row_i + col_i * col_i
                # self.dok_dict_ab[hash_val] = val

        hash_val = (row_i << 32) + col_i

        # cdef map[long, DATA_T].iterator it = self.dok_dict.find(hash_val)
        # if it != self.dok_dict.end():
        #     print "collision:", row_i, col_i

        cdef int digit_i = _calc_n_digits(hash_val)
        cdef int last_digit = hash_val % 10

        cdef vector[INDEX_T]* cur_rowcol_vector = self.rowcol_buckets[digit_i][last_digit]
        cdef vector[DATA_T]* cur_data_vector = self.data_buckets[digit_i][last_digit]

        cdef int it = 0, start_it = 0, end_it = cur_rowcol_vector.size() - 1, start_pos = 0, end_pos = cur_rowcol_vector.size()
        cdef int start_gap_end = end_it - start_it
        # print start_gap_end
        cdef INDEX_T cur_rowcol = 0

        cdef int i = 0, cur_pos = 0, insert_flag = 0, append_flag = 0

        # bisect left of cur_rowcol_vector, find insert position
        if cur_rowcol_vector.size() == 0: # empty bucket
            cur_rowcol_vector.push_back(hash_val)
            cur_data_vector.push_back(val)

        else:
            
            # left or right
            while 1:

                # if gap between start_it and end_it smaller than 5
                if start_gap_end <= 5:
                    for it in range(start_it, end_it+1):
                        cur_rowcol = deref(cur_rowcol_vector)[it]

                        if hash_val < cur_rowcol:
                            cur_pos = it
                            insert_flag = 1
                            break

                        elif hash_val == cur_rowcol:
                            cur_pos = it
                            insert_flag = 0
                            break

                        elif hash_val > cur_rowcol:
                            continue
                    
                    else:
                        append_flag = 1
                        cur_pos = end_it + 1
                    
                    break

                else:

                    # get middle it
                    it = (start_it + end_it) / 2

                    cur_rowcol = deref(cur_rowcol_vector)[it]

                    if hash_val == cur_rowcol : # find position
                        cur_pos = it
                        insert_flag = 0
                        break
                    
                    elif hash_val < cur_rowcol: # left
                        end_it = it

                    elif hash_val > cur_rowcol: # right
                        start_it = it
                    
                    start_gap_end = end_it - start_it

            # final process
            if get_flag:

                if insert_flag or append_flag: # no such item
                    return None

                else:
                    return deref(cur_data_vector)[cur_pos]
            
            else:

                if insert_flag:
                    cur_rowcol_vector.insert(cur_rowcol_vector.begin()+cur_pos, hash_val)
                    cur_data_vector.insert(cur_data_vector.begin()+cur_pos, val)

                    # if int(val) == 100:
                    #     print "insert", row_i, col_i, cur_pos, deref(cur_data_vector)[cur_pos]

                elif append_flag:
                    cur_rowcol_vector.push_back(hash_val)
                    cur_data_vector.push_back(val)

                    # if int(val) == 100:
                    #     print "append", row_i, col_i, cur_pos, deref(cur_data_vector)[cur_pos]

                else: # already in 

                    if incr_flag:
                        # deref(cur_data_vector)[cur_pos] += val # cant use this, due to cython compile bug
                        deref(cur_data_vector)[cur_pos] = deref(cur_data_vector)[cur_pos] + val
                        
                        # if int(val) == 100:
                        #     print "incr", row_i, col_i, cur_pos, deref(cur_data_vector)[cur_pos]

                    else:
                        deref(cur_data_vector)[cur_pos] = val


        # self.dok_dict[hash_val] = val

    def set(self, INDEX_T row_i, INDEX_T col_i, DATA_T val):
        return self._bisect_op(row_i, col_i, val, incr_flag=0)

    def get(self, INDEX_T row_i, INDEX_T col_i):
        return self._bisect_op(row_i, col_i, get_flag=1)

    def incr(self, INDEX_T row_i, INDEX_T col_i, DATA_T val):
        return self._bisect_op(row_i, col_i, val, incr_flag=1)

    def shrink_by_thres(self, DATA_T thres):

        cdef vector[INDEX_T]* cur_rowcol_vector
        cdef vector[DATA_T]* cur_data_vector
        cdef vector[INDEX_T].iterator rowcol_vector_iter
        cdef vector[DATA_T].iterator data_vector_iter

        cdef INDEX_T cur_rowcol = 0
        cdef DATA_T cur_data = 0

        cdef int i = 0, j = 0
        # ite all 
        for i in range(1, 21):
            for j in range(1, 10):
                cur_rowcol_vector = self.rowcol_buckets[i][j]
                cur_data_vector = self.data_buckets[i][j]

                # ite one 
                rowcol_vector_iter = cur_rowcol_vector.begin()
                data_vector_iter = cur_data_vector.begin()

                while rowcol_vector_iter != cur_rowcol_vector.end():
                    # cur_rowcol = deref(rowcol_vector_iter)
                    cur_data = deref(data_vector_iter)

                    if cur_data <= thres:
                        rowcol_vector_iter = cur_rowcol_vector.erase(rowcol_vector_iter)
                        data_vector_iter = cur_data_vector.erase(data_vector_iter)

                    else:
                        inc(rowcol_vector_iter)
                        inc(data_vector_iter)

    def del_symi_batch(self, set symi_set):

        # del stuff in SuzuMatrix is expensive, because we have to ite through all
        # symi is short for symmetric index, only work if matrix is symmetric, and symi is both row_i and col_i

        cdef cppset[INDEX_T] symi_cppset = symi_set
        cdef cppset[INDEX_T].iterator symi_cppset_end = symi_cppset.end()

        cdef vector[INDEX_T]* cur_rowcol_vector
        cdef vector[DATA_T]* cur_data_vector
        cdef vector[INDEX_T].iterator rowcol_vector_iter
        cdef vector[DATA_T].iterator data_vector_iter

        cdef INDEX_T cur_rowcol = 0
        cdef unsigned int cur_row = 0
        cdef unsigned int cur_col = 0

        cdef int i = 0, j = 0

        # ite all 
        for i in range(1, 21):
            for j in range(1, 10):
                cur_rowcol_vector = self.rowcol_buckets[i][j]
                cur_data_vector = self.data_buckets[i][j]

                # ite one 
                rowcol_vector_iter = cur_rowcol_vector.begin()
                data_vector_iter = cur_data_vector.begin()

                while rowcol_vector_iter != cur_rowcol_vector.end():
                    cur_rowcol = deref(rowcol_vector_iter)
                    cur_col = <unsigned int>cur_rowcol
                    cur_row = <unsigned int>(cur_rowcol >> 32)

                    if symi_cppset.find(cur_row) != symi_cppset_end or symi_cppset.find(cur_col) != symi_cppset_end: # cur_row should be del
                        rowcol_vector_iter = cur_rowcol_vector.erase(rowcol_vector_iter)
                        data_vector_iter = cur_data_vector.erase(data_vector_iter)

                    else:
                        inc(rowcol_vector_iter)
                        inc(data_vector_iter)

    def reset_symi_all(self, dict symi_mapping_dict):

        # cdef vector[INDEX_T]* new_rowcol_buckets[21][10]
        # cdef vector[DATA_T]* new_data_buckets[21][10]

        cdef unordered_map[unsigned int, unsigned int] symi_mapping = symi_mapping_dict
        cdef unordered_map[unsigned int, unsigned int].iterator symi_mapping_row_it, symi_mapping_col_it

        cdef vector[INDEX_T]* old_rowcol_buckets[21][10]
        cdef vector[DATA_T]* old_data_buckets[21][10]

        cdef int i = 0, j = 0
        for i in range(1, 21):
            for j in range(1, 10):
                old_rowcol_buckets[i][j] = self.rowcol_buckets[i][j]
                old_data_buckets[i][j] = self.data_buckets[i][j]

                # pre-allocated memory
                self.rowcol_buckets[i][j] = new vector[INDEX_T](old_rowcol_buckets[i][j].size())
                self.data_buckets[i][j] = new vector[DATA_T](old_data_buckets[i][j].size())

        cdef vector[INDEX_T]* cur_rowcol_vector
        cdef vector[DATA_T]* cur_data_vector
        cdef vector[INDEX_T].iterator rowcol_vector_iter
        cdef vector[DATA_T].iterator data_vector_iter

        cdef INDEX_T cur_rowcol = 0
        cdef DATA_T cur_data = 0
        cdef unsigned int cur_row = 0
        cdef unsigned int cur_col = 0

        # ite all and set all
        for i in range(1, 21):
            for j in range(1, 10):
                cur_rowcol_vector = old_rowcol_buckets[i][j]
                cur_data_vector = old_data_buckets[i][j]

                rowcol_vector_iter = cur_rowcol_vector.begin()
                data_vector_iter = cur_data_vector.begin()

                while rowcol_vector_iter != cur_rowcol_vector.end():

                    cur_rowcol = deref(rowcol_vector_iter)
                    cur_data = deref(data_vector_iter)
                    cur_col = <unsigned int>cur_rowcol
                    cur_row = <unsigned int>(cur_rowcol >> 32)

                    # update only when both row and col has mapping
                    symi_mapping_row_it = symi_mapping.find(cur_row)
                    symi_mapping_col_it = symi_mapping.find(cur_col)

                    if symi_mapping_row_it != symi_mapping.end() and symi_mapping_col_it != symi_mapping.end() :

                        # print "old symi", cur_row, cur_col

                        cur_row = deref(symi_mapping_row_it).second
                        cur_col = deref(symi_mapping_col_it).second

                        # print "new symi", cur_row, cur_col
                    
                    else:
                        # print "missing symi", cur_row, cur_col
                        pass

                    self.set(cur_row, cur_col, cur_data)

                    inc(rowcol_vector_iter)
                    inc(data_vector_iter)

                # clear old buckets, otherwise lead to memory leak
                del old_rowcol_buckets[i][j]
                del old_data_buckets[i][j]


    def status(self):
        cdef dict status_dict = {}

        cdef vector[INDEX_T]* cur_rowcol_vector
        cdef vector[DATA_T]* cur_data_vector

        cdef int i = 0, j = 0
        for i in range(1, 21):
            for j in range(1, 10):
                cur_rowcol_vector = self.rowcol_buckets[i][j]
                cur_data_vector = self.data_buckets[i][j]

                status_dict[(i,j)] = [cur_rowcol_vector.size(), cur_data_vector.size()]

        return status_dict

    def astuples(self):
        cdef vector[INDEX_T]* cur_rowcol_vector
        cdef vector[DATA_T]* cur_data_vector
        cdef vector[INDEX_T].iterator rowcol_vector_iter
        cdef vector[DATA_T].iterator data_vector_iter

        cdef INDEX_T cur_rowcol = 0
        cdef DATA_T cur_data = 0
        cdef unsigned int cur_row = 0
        cdef unsigned int cur_col = 0

        cdef list tuple_list = []

        # ite all and set all
        cdef int i = 0, j = 0
        for i in range(1, 21):
            for j in range(1, 10):
                cur_rowcol_vector = self.rowcol_buckets[i][j]
                cur_data_vector = self.data_buckets[i][j]

                rowcol_vector_iter = cur_rowcol_vector.begin()
                data_vector_iter = cur_data_vector.begin()

                while rowcol_vector_iter != cur_rowcol_vector.end():

                    cur_rowcol = deref(rowcol_vector_iter)
                    cur_data = deref(data_vector_iter)
                    cur_col = <unsigned int>cur_rowcol
                    cur_row = <unsigned int>(cur_rowcol >> 32)

                    if cur_data != 0:
                        tuple_list.append( (cur_row,cur_col,cur_data) )

                    inc(rowcol_vector_iter)
                    inc(data_vector_iter)

        return tuple_list

    def tostring(self):
        cdef list tuple_list = self.astuples(), ss_list = []
        for cur_row, cur_col, cur_data in sorted(tuple_list, key=lambda x: x[0]):
            ss_list.append(str(cur_row)+","+str(cur_col)+":"+str(cur_data))

        return "\n".join(ss_list)

    def __repr__(self):
        return self.tostring()

    def __str__(self):
        return self.tostring()





cdef class WordCoocCounter:

    cdef public dict word2id_mapping
    cdef public dict word_cooc_dok
    cdef public SuzuMatrix word_cooc_suzu
    cdef public dict id2word_mapping

    cdef public int n_total_word
    cdef public object debug_flag

    def __init__(self, debug_flag=False):
        self.word2id_mapping = {}
        self.word_cooc_dok = {}
        self.id2word_mapping = {}
        self.n_total_word = 0

        self.word_cooc_suzu = SuzuMatrix()

        self.debug_flag = debug_flag

    # property id2word_mapping:
    #     "A doc string can go here."

    #     def __get__(self):
    #         # This is called when the property is read.
    #         return self.id2word_mapping

    # property word_cooc_dok:
    #     "A doc string can go here."

    #     def __get__(self):
    #         # This is called when the property is read.
    #         return self.word_cooc_dok

    def count_word_freq(self):
        cdef int n_total_word = 0
        cdef list val 

        for val in self.id2word_mapping.itervalues():
            n_total_word += val[-1]

        self.n_total_word = n_total_word
        return n_total_word

    def fit_doc_words_batch(self, doc_words_list):
        """
        word2id_mapping: {
            word: [id, n_freq]
        }
        """

        cdef dict word2id_mapping = self.word2id_mapping
        cdef SuzuMatrix word_cooc_suzu = self.word_cooc_suzu
        cdef dict id2word_mapping = self.id2word_mapping

        # cdef list doc_words = []
        cdef tuple doc_words = ()
        cdef str word = ""
        cdef set word_set = set()
        cdef int word_id = 0
        cdef int cword_id = 0
        cdef list word_id_list = []

        cdef int i = 0
        cdef int word_id_list_length = 0
        cdef dict word_dict = {}
        cdef int temp_word_id = 0
        cdef int big_word_id = 0
        cdef int small_word_id = 0
        
        for doc_words_x in doc_words_list:

            if type(doc_words_x) == str or type(doc_words_x) == unicode:
                doc_words = tuple(doc_words_x.split())
            else:
                doc_words = tuple(doc_words_x)

            # run two passes inside the corpus iterate for memory usage
            
            word_set = set()

            for word in doc_words:
                if word in word2id_mapping:
                    word_id = word2id_mapping[word]
                    # word2id_mapping[word][1] += 1
                    id2word_mapping[word_id][1] += 1
                else:
                    word_id = len(word2id_mapping)
                    # print word_id
                    word2id_mapping[word] = word_id
                    id2word_mapping[word_id] = [word, 1]
                    
                word_set.add(word_id)

            word_id_list = list(word_set)
            word_id_list_length = len(word_id_list)
            # print "word_id_list_length:", word_id_list_length

            for i in range(0, word_id_list_length - 1):
                word_id = word_id_list[i]

                for j in range(i+1, word_id_list_length):
                    cword_id = word_id_list[j]
                    
                    # keep order and upsert 
                    if word_id > cword_id:
                        # swap
                        big_word_id = word_id
                        small_word_id = cword_id
                    else:
                        big_word_id = cword_id
                        small_word_id = word_id

                    word_cooc_suzu.incr(small_word_id, big_word_id, 1)

    def filter_words(self, n_top=1.0, int min_freq=-1, int min_cooc=1, auto_min_freq=True):
        # cdef dict word2id_mapping = self.word2id_mapping
        cdef dict id2word_mapping = self.id2word_mapping
        # cdef dict word_cooc_dok = self.word_cooc_dok
        cdef SuzuMatrix word_cooc_suzu = self.word_cooc_suzu

        cdef list sorted_id2word_mapping = []
        cdef int word_id = 0
        cdef int fword_id = 0
        cdef int cword_id = 0
        cdef dict word_dict = {}

        cdef int n_total_word = len(id2word_mapping)
        cdef list filter_word_id_list = []
        cdef set filter_word_id_set = set()
        cdef list empty_word_id_list = []
        cdef set empty_word_id_set = set()
        cdef set preserved_word_id_set = set()
        
        cdef list word_id_list = []
        cdef int n_cooc = 0
        cdef int n_freq = 0

        assert n_top != 0

        if "." in str(n_top):
            n_top = int(n_total_word * n_top)

        if n_top < 0:
            n_top = n_total_word + n_top

        # auto setting min_freq by long-tail theory
        if auto_min_freq:
            pass

        if n_total_word > n_top or min_freq > 0:
            sorted_id2word_mapping = sorted(id2word_mapping.items(), key=lambda x:x[1][1])

            # del word by n_top
            for i in range(0, n_total_word - n_top):
                word_id = sorted_id2word_mapping[i][0]
                # filter_word_id_list.append(word_id)
                filter_word_id_set.add(word_id)

                # delete word from both 
                del id2word_mapping[word_id]

            # del word by min_freq
            for i in range(0, len(sorted_id2word_mapping)):
                n_freq = sorted_id2word_mapping[i][1][1]
                if n_freq >= min_freq:
                    break

                word_id = sorted_id2word_mapping[i][0]
                filter_word_id_set.add(word_id)

                # delete word from both 
                try:
                    del id2word_mapping[word_id]
                except KeyError:
                    pass

        # print filter_word_id_listmin_cooc
        if self.debug_flag:
            print "inside id2word_mapping:", id2word_mapping
            print "inside word_cooc_suzu:", word_cooc_suzu

        # remove filter_word and min_cooc in cooc_dok
        word_cooc_suzu.del_symi_batch(filter_word_id_set)
        preserved_word_id_set = set(id2word_mapping.keys())

        if self.debug_flag:
            print "inside preserved_word_id_set:", preserved_word_id_set
        
        # reset word_id, 
        cdef dict word2id_mapping = self.word2id_mapping
        cdef dict new_id2word_mapping = {}
        cdef dict old_id2new_id_mapping = {}
        # cdef dict new_word_cooc_dok = {}
        cdef dict new_word2id_mapping = {}
        cdef int new_word_id = 0
        cdef dict new_word_dict = {}
        cdef str word = ""

        # new id2word_mapping
        for new_word_id,word_id in enumerate(preserved_word_id_set):
            new_id2word_mapping[new_word_id] = id2word_mapping[word_id]
            old_id2new_id_mapping[word_id] = new_word_id

        self.word_cooc_suzu.reset_symi_all(old_id2new_id_mapping)

        for word, word_id in word2id_mapping.iteritems():
            try:
                new_word2id_mapping[word] = old_id2new_id_mapping[word_id]
            except KeyError:
                continue

        self.id2word_mapping = new_id2word_mapping
        # self.word_cooc_dok = new_word_cooc_dok
        self.word2id_mapping = new_word2id_mapping

    def gen_word_coo_mtx(self):
        cdef dict word_cooc_dok = self.word_cooc_dok
        cdef list row_list = []
        cdef list col_list = []
        cdef list data_list = []
        
        cdef int word_id, n_cooc, cword_id
        cdef dict word_dict

        for word_id, word_dict in word_cooc_dok.iteritems():
            for cword_id, n_cooc in word_dict.iteritems():
                row_list.append(word_id)
                col_list.append(cword_id)
                data_list.append(n_cooc)

        cdef np.ndarray[int, ndim=1] row_arr = np.asarray(row_list, dtype=np.int32)
        cdef np.ndarray[int, ndim=1] col_arr = np.asarray(col_list, dtype=np.int32)
        cdef np.ndarray[double, ndim=1] data_arr = np.asarray(data_list, dtype=np.DATA_T64)

        cdef int n_words = len(self.id2word_mapping)
        coo_mtx = sp.coo_matrix((data_arr, (row_arr, col_arr)), shape=(n_words, n_words))

        return coo_mtx





        



