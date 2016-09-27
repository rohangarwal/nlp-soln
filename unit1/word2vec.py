from gensim.models import Word2Vec

if __name__ == '__main__':

    ex=[]
    for line in open('corpus.txt'):
        ex.append(line)

    vocab = [s.encode('utf-8').split() for s in ex]
    model = Word2Vec(vocab,min_count=176)
    model.save("model.txt")
