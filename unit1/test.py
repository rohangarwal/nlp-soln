from gensim.models import Word2Vec

if __name__ == '__main__':
    model = Word2Vec.load("model.txt")

    print model.similarity('govt', 'government')
    print model.similarity('rahul', 'sonia')
    print model.similarity('modi', 'modi')
    print model.similarity('india', 'uae')
    print model.similarity('taliban', 'terrorism')
    print model.similarity('modi', 'pm')
    print model.similarity('namo', 'modi')
    print model.similarity('muslim', 'arab')
    print model.similarity('dubai', 'city')
    print model.similarity('congress', 'bjp')