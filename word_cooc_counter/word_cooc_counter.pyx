


import numpy as np
cimport numpy as np
import scipy.sparse as sp


cdef class WordCoocCounter:

    cdef public dict word2id_mapping
    cdef public dict word_cooc_dok
    cdef public dict id2word_mapping

    def __init__(self):
        self.word2id_mapping = {}
        self.word_cooc_dok = {}
        self.id2word_mapping = {}

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

    def get_word2id_mapping(self):
        return self.word2id_mapping

    def get_word_cook_dok(self):
        return self.word_cooc_dok

    def word2ids(self):
        # first pass to gen dictionary, and also count word_freq
        return

    def word_cooc_count(self, doc_str_list, mode=0):
        # count frequent word cooc, 

        """
        mode list:
        0: 
        """
        return

    def fit_doc_str(self, list doc_str_list):
        return

    def fit_doc_words_batch(self, doc_words_list):
        """
        word2id_mapping: {
            word: [id, n_freq]
        }
        """

        cdef dict word2id_mapping = self.word2id_mapping
        cdef dict word_cooc_dok = self.word_cooc_dok
        cdef dict id2word_mapping = self.id2word_mapping

        cdef list doc_words = []
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
        
        for doc_words in doc_words_list:

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

    def filter_words(self, n_top=1.0, int min_freq=-1, int min_cooc=1):
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

        cdef np.ndarray[int, ndim=1] row_arr = np.asarray(row_list)
        cdef np.ndarray[int, ndim=1] col_arr = np.asarray(col_list)
        cdef np.ndarray[double, ndim=1] data_arr = np.asarray(data_list)

        cdef int n_words = len(self.id2word_mapping)
        sp.coo_matrix((data_arr, (row_arr, col_arr)), shape=(n_words, n_words))







        



