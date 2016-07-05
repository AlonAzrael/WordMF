# coding:utf-8


from word_cooc_counter import WordCoocCounter, SuzuMatrix
from jiebax import JiebaX
JIEBAX = JiebaX()


from time import sleep
import os, re
import cPickle as pickle
from collections import defaultdict

from scipy.sparse import dok_matrix, lil_matrix
import numpy as  np

from sortedcontainers import SortedDict
from pprint import pprint


def load_docs(filepath):
    return open(filepath, "r").xreadlines()
    # return open(filepath, "r").readlines()[:100]


def docs2sents(docs, spliter=u"。"):

    sents_all = []
    
    for doc in docs:
        if type(doc) == str:
            doc = unicode(doc, "utf-8", "ignore")
        
        sents = doc.split(u"。")
        sents_all += sents

    return sents_all


def clean_sents(sents):

    clean_sents = []
    
    for sent in sents:
        if type(sent) == str:
            sent = unicode(sent, "utf-8", "ignore")
        
        clean_sent = re.sub( ur"[^\u4e00-\u9fa5\n]+", u" ", sent, flags=re.UNICODE)
        clean_sents.append(clean_sent)

    return clean_sents


def cut_sents(sents):

    csent_list = []
    
    for sent in sents:
        words = JIEBAX.posseg_filter( sent, startswith_list=["n"] )
        csent = " ".join(words)
        csent_list.append(csent)

    return csent_list


def docs2wordslist(filepath):
    docs = load_docs(filepath)
    sents = docs2sents(docs)

    # raw_input("before del docs")
    # del docs
    # raw_input("after del docs")

    sents = clean_sents(sents)
    sents = cut_sents(sents)

    print "n_sents:", len(sents)

    with open("/".join(filepath.split("/")[:-1]+["wiki_cn_sent_words"]), "w") as F:
        F.write("\n".join(sents))

    return sents


def wordslist2coomat(filepath):
    wordslist = []

    with open(filepath, "r") as F:
        # for line in F.xreadlines():
        #     words = line.split()
        #     wordslist.append(words)
        wordslist = F.xreadlines()

        wcc = WordCoocCounter(debug_flag=False)
        wcc.fit_doc_words_batch(wordslist)
    
    raw_input("pause")
    return

    wcc.filter_words(min_freq=20)

    coo_mtx = wcc.gen_word_coo_mtx()
    # print coo_mtx
    with open("./word_coo_mtx.pkl", "wb") as F:
        pickle.dump(coo_mtx, F, -1)


def test_word_dict_memory_usage(filepath):
    wordslist = []

    word_counter = {}

    with open(filepath, "r") as F:
        for line in F.xreadlines():
            words = line.split()
            # wordslist.append(words)

            for w in words:
                try:
                    word_counter[w] += 1
                except KeyError:
                    word_counter[w] = 1

                # word_counter[w] += 1

    print len(word_counter)
    raw_input("pause")


def test_dod_memory_usage():

    height = 50; width = 6
    s, b = 0, 0
    
    # x = {i:{j:1 for j in xrange(width)} for i in xrange(height)}
    # x = {i: SortedDict({j:1 for j in xrange(width)}) for i in xrange(height)}
    # x = [ zip(*[(j,1) for j in xrange(width)] ) for i in xrange(height)]
    
    # x = {}
    # for i in xrange(height):
    #     for j in xrange(width):
    #         if i>=j:
    #             hash_val = i*i+i+j
    #         else:
    #             hash_val = i+j*j
    #         # hash_val = (i+j)*(i+j+1)*0.5+j

    #         # if hash_val not in x:
    #         #     x[hash_val] = 1
    #         # else:
    #         #     print i,j

    #         x[hash_val] = 1

    # print len(x)

    # x = dok_matrix((height, width), dtype=np.int32)
    # for i in xrange(height):
    #     for j in xrange(width):
    #         x[i,j] = 1

    # x = lil_matrix((height, width), dtype=np.int32)
    # for i in xrange(height):
    #     for j in xrange(width):
    #         x[i,j] = 1

    x = SuzuMatrix(height, width)
    # print x.calc_n_digits(2**64-1)
    new_symi_mapping = {}
    for i in xrange(height):
        for j in xrange(width):

            if i >= j:
                s, b = j, i
            else:
                s, b = i, j

            new_symi_mapping[s] = s+1
            new_symi_mapping[b] = b+1
            
            # s, b = i, j
            
            x.incr(s, b, s*b)
    
    # pprint(x.status())
    print x.get(5, 20)
    print x.get(20, 5)
    print x.get(5, 5)
    print x.get(20, 20)
    x.reset_symi_all(new_symi_mapping)
    x.del_symi_batch(set([5, 20]))

    print "after clean"
    print x.get(5, 20)
    print x.get(6, 21)
    print x.get(20, 5)
    print x.get(5, 5)
    print x.get(20, 20)

    print x

    # np.ones((height, width), dtype=np.int32)

    raw_input("pause")




if __name__ == '__main__':
    # docs2wordslist("/home/aaronyin/TheCoverProject/Wikipedia_Corpus/wiki_cn_lines")
    # wordslist2coomat("/home/aaronyin/TheCoverProject/Wikipedia_Corpus/wiki_cn_sent_words")

    # test_memory_usage("/home/aaronyin/TheCoverProject/Wikipedia_Corpus/wiki_cn_sent_words")
    test_dod_memory_usage()

