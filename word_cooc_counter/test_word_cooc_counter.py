# coding:utf-8

import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs":np.get_include()})

from word_cooc_counter import WordCoocCounter


def test_word_cooc_counter():
    doc_words_list = [
        "你好，再见，",
        "你好，世界，",
        "世界，地球，",
        "再见，地球，",
        "再见，世界，",
        "欧洲，世界，地球，",
        "你好，欧洲",
        "你好，欧洲",
        "你好，宇宙",
        "你好，太空",
        "欧洲，宇宙，宇航",
    ]
    
    spliter = "，"
    
    for i,s in enumerate(doc_words_list):
        if s.endswith(spliter):
            doc_words_list[i] = s.split(spliter)[:-1]
        else:
            doc_words_list[i] = s.split(spliter)

    # print doc_words_list

    wcc = WordCoocCounter()
    wcc.fit_doc_words_batch(doc_words_list)

    # print wcc.word_cooc_dok
    print "=== state ==="
    for k,v in wcc.id2word_mapping.iteritems():
        print k, v[0], v[1]
    print wcc.word_cooc_dok
    print "=== end state ==="

    wcc.filter_words(n_top=1.0, min_freq=2)

    print "=== state ==="
    for k,v in wcc.id2word_mapping.iteritems():
        print k, v[0], v[1]
    print wcc.word_cooc_dok
    print "=== state ==="
    # for k,v in wcc.id2word_mapping.iteritems():
    #     print k, v[0]






if __name__ == '__main__':
    test_word_cooc_counter()


