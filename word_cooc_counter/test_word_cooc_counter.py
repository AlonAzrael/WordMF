# coding:utf-8



from __init__ import WordCoocCounter, LodMatrix


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

    wcc = WordCoocCounter(debug_flag=True)
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

    print wcc.gen_word_coo_mtx()



def test_lod_matrix():
    x = LodMatrix()
    for i in xrange(50000):
        for j in xrange(500):
            x.set(i,j, 1)

    raw_input("pause")



if __name__ == '__main__':
    # test_word_cooc_counter()
    test_lod_matrix()


