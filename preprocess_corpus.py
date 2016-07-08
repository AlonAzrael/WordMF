# coding:utf-8


from word_cooc_counter import WordCoocCounter, CorpusReader
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

    corpus_reader = CorpusReader(filepath)
    # manually add words 
    sents_s = """
    V字仇杀队 政府 民主 革命 独裁 反独裁 英雄 自由 信仰 社会 
    帝企鹅日记 南极 寒冷 生存 纪录 环境 动物 生态 环保 
    战争之王 政治 人性 战争 独裁 犯罪 黑帮 军火
    断背山 爱情 同性恋 乡村 羁绊
    罪恶之城 犯罪 黑帮 暴力 黑暗 动作
    傲慢与偏见 爱情 乡村 英伦 
    电锯惊魂 恐怖 惊悚 暴力 死亡 动作
    当幸福来敲门 城市 生存 亲情 父爱 奋斗 幸福 人生价值 生活
    荒野生存 生存 环境 流浪 自由 人生价值 生活
    致命魔术 魔术 悬疑 友情 羁绊
    谍影重重3 间谍 暴力 动作 悬疑
    美食总动员 美食 动画 大厨 
    朱诺 青春 校园 成长 怀孕 爱情 
    机器人总动员 机器人 太空 科幻 友情 动画 羁绊
    本杰明巴顿奇事 科幻 人生价值 二战 爱情 生活
    冰雪奇缘 动画 公主 王子 爱情 童话 女王 皇后 姐妹 亲情 拯救 牺牲 善良 羁绊 中世纪 冰 雪 
    泰坦尼克号 爱情 灾难 海 海洋 轮船 牺牲 爱 奥斯卡 拯救 羁绊
    蝙蝠侠之黑暗骑士 英雄 黑暗 保护 拯救 牺牲 科技 技术 装备 打击犯罪 
    搏击俱乐部 人生价值 自由 悬疑 疯狂 激情 不羁 精神
    海豚湾 环境 动物 环保 生态 生存 海豚 海 海洋
    忠犬八公的故事 动物 狗 善良 羁绊
    阿凡达 环保 环境 科幻 太空 保护 英雄 星球 
    盗梦空间 科幻 幻想 拯救 梦境 精神 
    玛丽和马克思 友情 羁绊 动画 
    月球 科幻 太空 星球 宇航员 克隆人 克隆 伦理 道德 
    禁闭岛 悬疑 精神 侦探 
    源代码 科幻 爱情 程序 计算机 代码 人工智能 智能
    霍比特人之意外之旅 英雄 奇幻 神秘 探险 冒险 中世纪 古代 骑士 
    复仇者联盟 英雄 科幻 宇宙 动作 超能力 
    """
    sents = [sent.strip() for sent in sents_s.split("\n") if len(sent.strip())>3]
    # pprint(sents)
    keywords = [sent.split()[0] for sent in sents]
    for _ in xrange(2000):
        corpus_reader.append_sents(sents)

    print "init wcc ..."
    wcc = WordCoocCounter(debug_flag=False)
    wcc.fit_corpus(corpus_reader, min_word_freq=20)

    # for k,v in wcc.get_dictionary().iteritems():
    #     print k, v[0], v[1]
    # print wcc.tocoo()
    
    # raw_input("pause")
    # return

    keyword_ids = []
    for word in keywords:
        word_id = wcc._wcc_dict.get(word)[0]
        keyword_ids.append(word_id)

    with open("./__X_data/keyword_ids", "w") as F:
        F.write(repr(keyword_ids))

    with open("./__X_data/wcc_dict", "w") as F:
        F.write(wcc._wcc_dict.asid2words().tostring())

    coo_mtx = wcc.tocoo()
    # print coo_mtx
    with open("./__X_data/word_coo_mtx.pkl", "wb") as F:
        pickle.dump(coo_mtx, F, -1)



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
    wordslist2coomat("/home/aaronyin/TheCoverProject/Wikipedia_Corpus/wiki_cn_doc_words_nx")

    # test_memory_usage("/home/aaronyin/TheCoverProject/Wikipedia_Corpus/wiki_cn_sent_words")
    # test_dod_memory_usage()

