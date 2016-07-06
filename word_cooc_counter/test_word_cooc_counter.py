# coding:utf-8



from __init__ import AoaMatrix, WordCoocCounter


def test_word_cooc_counter():
    doc_words_list = [
        "你好，再见，",
        "你好，世界，",
        "你好，世界，",
        "世界，地球，",
        "再见，地球，",
        "再见，世界，",
        "你好，欧洲",
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
    wcc.fit_corpus(doc_words_list, min_word_freq=2)

    # print wcc.word_cooc_dok
    print "=== state ==="
    for k,v in wcc.get_dictionary().iteritems():
        print k, v[0], v[1]
    print wcc.get_aoa_mtx()
    print wcc.tocoo()
    print "=== end state ==="

    # print "=== state ==="
    # for k,v in wcc.id2word_mapping.iteritems():
    #     print k, v[0], v[1]
    # print wcc.word_cooc_suzu
    # print "=== state ==="

    # print wcc.gen_word_coo_mtx()



def test_aoa_matrix():
    height = 50000; width = 500
    aoa = AoaMatrix(height, width)

    raw_input("pause")


def test_wcc_matrix(mat_class):
    height = 50; width = 6

    print "init mat ..."
    x = mat_class(height, width)
    # print x.calc_n_digits(2**64-1)
    new_symi_mapping = {}
    for i in xrange(height):
        for j in xrange(width):

            if i > j:
                s, b = j, i
            elif i < j:
                s, b = i, j
            else:
                continue

            new_symi_mapping[s] = s+1
            new_symi_mapping[b] = b+1
            
            # s, b = i, j
            
            # print "incr", s, b
            x.incr(s, b, s*b)
    
    print "init mat done"
    # pprint(x.status())
    print x.get(5, 20)
    print x.get(20, 5)
    print x.get(5, 5)
    print x.get(20, 20)

    print "after clean"
    print x.get(5, 20)
    print x.get(6, 21)
    print x.get(20, 5)
    print x.get(5, 5)
    print x.get(20, 20)

    print x.tocoo()
    # print x

    raw_input("pause")


def test_str_unicode_memory_usage():
    word = "你好"
    n_words = 1000000*10
    dd = {}
    
    for i in xrange(n_words):
        dd[word + unicode(i)] = i

    raw_input("pause")



if __name__ == '__main__':
    test_word_cooc_counter()
    # test_wcc_matrix(AoaMatrix)
    # test_aoa_matrix()

    # test_str_unicode_memory_usage()


