
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False


import numpy as np
cimport numpy as np
import scipy.sparse as sp

from libc.stdlib cimport malloc, free

from libcpp.map cimport map
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc



ctypedef map[int, double]* map_pointer

cdef class LodMatrix:

    # in name of list of dict
    # still too much memory
    
    # cdef map_pointer rows_array
    cdef int n_rows
    # cdef int max_rows_size
    # cdef double expand_rate

    cdef vector[map[int, double]] rows_vector

    def __cinit__(self, int n_rows=100):

        # self.expand_rate = 1.5

        # self.rows_array = <map_pointer>malloc(n_rows*sizeof(map_pointer))
        # for i in range(n_rows):
        #     self.rows_array[i] = new map[int, double]()

        self.n_rows = n_rows
        # self.max_rows_size = n_rows

    def __dealloc__(self):
        # for i in range(self.n_rows):
        #     del self.rows_array[i]

        # free(self.rows_array)

        self.rows_vector.clear()

    def set(self, int row_i, int col_i, double val):
        # will auto epxand rows_array 
        # cdef int n_rows = self.n_rows
        # cdef int new_rows_size = self.max_rows_size
        # cdef map_pointer rows_array = self.rows_array
        # cdef map_pointer new_rows_array

        # cdef int i = 0

        # if row_i >= self.max_rows_size:
        #     # epxand 
        #     new_rows_size = n_rows*self.expand_rate

        #     new_rows_array = <map_pointer>malloc(new_rows_size*sizeof(map_pointer))
        #     for i in range(n_rows):
        #         new_rows_array[i] = rows_array[i]

        #     for i in range(n_rows, new_rows_size):
        #         new_rows_array[i] = new map[int, double]()

        #     self.rows_array = new_rows_array
        #     self.max_rows_size = new_rows_size

        #     # free the old
        #     free(rows_array)

        # deref(self.rows_array[row_i])[col_i] = val

        if row_i >= self.rows_vector.size():
            self.rows_vector.resize(row_i+1)

        self.rows_vector[row_i][col_i] = val

        # if row_i >= self.n_rows: # note for array bound
        #     self.n_rows = row_i + 1

    def get(self, int row_i, int col_i):
        cdef double val = 0

        if row_i >= self.rows_vector.size():
            return None

        cdef map[int, double] col_dict = self.rows_vector[row_i]
        cdef map[int, double].iterator it = col_dict.find(col_i)
        
        if it == col_dict.end():
            return None
        else:
            val = deref(it).second
            return val

    def incr(self, int row_i, int col_i):
        cdef map[int, double] col_dict
        cdef map[int, double].iterator it
        
        if row_i >= self.rows_vector.size():
            self.rows_vector.resize(row_i)

            self.rows_vector[row_i][col_i] = 1

        else:
            col_dict = self.rows_vector[row_i]
            it = col_dict.find(col_i)
            if it == col_dict.end():
                col_dict[col_i] = 1
            else:
                col_dict[col_i] += 1
                # self.rows_vector[row_i][col_i] += 1

    def contains(self, int row_i, int col_i=-1):
        return



# cdef inline void resize_vector(vector[double] sas) nogil:



cdef class AoaMatrix:

    # in name of array of array
    cdef vector[vector[double]] rows_vector

    def __cinit__(self, int height, int width):
        pass

    def set(self, int row_i, int col_i, double val):
        if row_i >= self.rows_vector.size():
            self.rows_vector.resize(row_i+1)

        cdef vector[double] col_vector = self.rows_vector[row_i]

        if col_i >= col_vector.size():
            col_vector.resize(col_i+1)

        # col_vector[] = 
        self.rows_vector[row_i][col_i] = val

    def get(self, int row_i, int col_i):
        return

    def incr(self, int row_i, int col_i):
        return

    def contains(self, int row_i, int col_i=-1):
        return



cdef class SzudMatrix:

    cdef map[long, double] dok_dict
    cdef map[int, float] dok_dict_ab
    cdef map[int, float] dok_dict_ba

    cdef long* indices
    cdef double* datas

    def __cinit__(self, n_rows=10, n_cols=10):
        self.indices = <long*>malloc(n_rows*n_cols*sizeof(long))
        self.datas = <double*>malloc(n_rows*n_cols*sizeof(double))

        cdef int i = 0
        for i in range(n_rows*n_cols):
            self.indices[i] = i
            self.datas[i] = 1.0

        pass

    def set(self, long row_i, long col_i, double val):
        cdef long hash_val = 0

        if 0:
            if row_i >= col_i:
                hash_val = row_i * row_i + row_i + col_i
                # self.dok_dict_ab[hash_val] = val
            else: # mostly use this
                hash_val = row_i + col_i * col_i
                # self.dok_dict_ab[hash_val] = val

        hash_val = (row_i << 32) + col_i

        # cdef map[long, float].iterator it = self.dok_dict.find(hash_val)
        # if it != self.dok_dict.end():
        #     print "collision:", row_i, col_i

        self.dok_dict[hash_val] = val

    def get(self, int row_i, int col_i):

        pass




cdef class WordCoocCounter:

    cdef public dict word2id_mapping
    cdef public dict word_cooc_dok
    cdef public dict id2word_mapping
    cdef public int n_total_word
    cdef public object debug_flag

    def __init__(self, debug_flag=False):
        self.word2id_mapping = {}
        self.word_cooc_dok = {}
        self.id2word_mapping = {}
        self.n_total_word = 0

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
        cdef dict word_cooc_dok = self.word_cooc_dok # word_cooc_dok cant be used due to crazy-expensive memory usage
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

                    try:
                        word_dict = word_cooc_dok[small_word_id]
                        try:
                            word_dict[big_word_id] += 1
                        except KeyError:
                            # print word_cooc_dok
                            word_dict[big_word_id] = 1
                    except KeyError:
                        word_cooc_dok[small_word_id] = {big_word_id:1}

        # manually free memory
        if 0:
            for word_id,word_dict in word_cooc_dok.items():
                word_dict.clear()

            self.word_cooc_dok.clear()
            word_cooc_dok.clear()
            
            id2word_mapping.clear()
            self.id2word_mapping.clear()
            
            word2id_mapping.clear()
            self.word2id_mapping.clear()

    def filter_words(self, n_top=1.0, int min_freq=-1, int min_cooc=1, auto_min_freq=True):
        # cdef dict word2id_mapping = self.word2id_mapping
        cdef dict id2word_mapping = self.id2word_mapping
        cdef dict word_cooc_dok = self.word_cooc_dok

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
            sorted_id2word_mapping

            # del word by n_top
            for i in range(0, n_total_word - n_top):
                word_id = sorted_id2word_mapping[i][0]
                # filter_word_id_list.append(word_id)
                filter_word_id_set.add(word_id)

                # delete word from both 
                del id2word_mapping[word_id]
                try:
                    del word_cooc_dok[word_id]
                except KeyError:
                    continue

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
                try:
                    del word_cooc_dok[word_id]
                except KeyError:
                    pass

        # print filter_word_id_listmin_cooc
        if self.debug_flag:
            print "inside id2word_mapping:", id2word_mapping
            print "inside word_cooc_dok:", word_cooc_dok

        # remove filter_word and min_cooc in cooc_dok
        for word_id, word_dict in word_cooc_dok.iteritems():

            # word_id_list = []
            for cword_id, n_cooc in word_dict.items():
                if cword_id in filter_word_id_set:
                    del word_dict[cword_id]

                # if n_cooc <= min_cooc:

                else:
                    preserved_word_id_set.add(cword_id)

            if len(word_dict) == 0:
                # empty_word_id_set.add(word_id)
                empty_word_id_list.append(word_id)
            else:
                preserved_word_id_set.add(word_id)

        for word_id in empty_word_id_list:
            # word_cooc_dok only have small word_id, so...
            del word_cooc_dok[word_id]

        if self.debug_flag:
            print "inside preserved_word_id_set:", preserved_word_id_set
        
        # reset word_id, 
        cdef dict word2id_mapping = self.word2id_mapping
        cdef dict new_id2word_mapping = {}
        cdef dict old_id2new_id_mapping = {}
        cdef dict new_word_cooc_dok = {}
        cdef dict new_word2id_mapping = {}
        cdef int new_word_id = 0
        cdef dict new_word_dict = {}
        cdef str word = ""

        for new_word_id,word_id in enumerate(preserved_word_id_set):
            new_id2word_mapping[new_word_id] = id2word_mapping[word_id]
            old_id2new_id_mapping[word_id] = new_word_id

        for word_id,word_dict in word_cooc_dok.iteritems():
            new_word_dict = {}
            for cword_id, n_cooc in word_dict.iteritems():
                new_word_dict[old_id2new_id_mapping[cword_id]] = n_cooc
            
            new_word_cooc_dok[old_id2new_id_mapping[word_id]] = new_word_dict

        for word, word_id in word2id_mapping.iteritems():
            try:
                new_word2id_mapping[word] = old_id2new_id_mapping[word_id]
            except KeyError:
                continue

        self.id2word_mapping = new_id2word_mapping
        self.word_cooc_dok = new_word_cooc_dok
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
        cdef np.ndarray[double, ndim=1] data_arr = np.asarray(data_list, dtype=np.float64)

        cdef int n_words = len(self.id2word_mapping)
        coo_mtx = sp.coo_matrix((data_arr, (row_arr, col_arr)), shape=(n_words, n_words))

        return coo_mtx





        



