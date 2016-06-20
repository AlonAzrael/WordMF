

from wordemb.wordemb_exp import GloveWrapper, test_most_similar, test_word_cluster



def load_corpus():
    with open("../Wikipedia_Corpus/wiki_cn_doc_words_nx", "r") as F:
        lines = F.readlines()

    docs = [line.split() for line in lines]
    print "load_corpus OK"

    return docs


def train(docs):
    model = GloveWrapper(n_features=200, name="./__X_model/glove.model")
    model.fit(docs).save()


def cluster_word():
    model = GloveWrapper(n_features=200, name="./__X_model/glove.model")
    model.load()



if __name__ == '__main__':
    # docs = load_corpus()
    # train(docs)

    modelwrapper = GloveWrapper(n_features=200, name="./__X_model/glove.model").load()
    # test_most_similar(modelwrapper)
    test_word_cluster(modelwrapper, filepath="./word_cluster.txt", n_clusters=100)





