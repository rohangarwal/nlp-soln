import json
import nltk
from nltk.corpus import stopwords
import sys
import pickle
reload(sys)
sys.setdefaultencoding('utf-8')

def Preprocess(t):
    '''
        Everything reduced to lower case for ease of processing
    '''
    t = t.lower()

    # Specific case to remove <'> so as to tag properly
    t = t.replace('\'','')

    return t

def Postag(t):
    '''
        Text Segmentation and Tagging
    '''
    sentences = nltk.sent_tokenize(t)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

    # Attaching POS tags
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]

    # Flattening list of lists
    pos_tagged = [item for sublist in tagged_sentences for item in sublist]

    return pos_tagged

def Chunking(t):
    '''
        Chunking is grouping tagged words as phrases
    '''
    # Tag pattern to identify dishes
    pattern = '''Senti : {<NN.*>|<VB.*>|<JJ.*>}'''

    chunk_rule = nltk.RegexpParser(pattern)
    tree = chunk_rule.parse(t)

    return tree

def Treeparse(tree):
    '''
        To parse chunk tree
    '''
    foods = []
    for subtree in tree.subtrees():
        if subtree.label() == 'Senti':
            foods.append(' '.join([str(child[0]) for child in subtree]))

    return foods

def Analyse(review):
    processed = Preprocess(str(review))
    tagged = Postag(processed)
    chunked = Chunking(tagged)
    parsed = set(Treeparse(chunked))

    return parsed


if __name__ == '__main__':
    file = open('pickles/train.pkl','rb')
    ls = pickle.load(file)
    comp = set()
    disp = set()
    mice = set()
    for tweets in ls:
        parsed = Analyse(tweets[0])
        if tweets[1] == 'miscellaneous':
            mice |= parsed
        elif tweets[1] == 'displeasure':
            disp |= parsed
        else:
            comp |= parsed

    dic = {}
    dic['compliment'] = comp
    dic['miscellaneous'] = mice
    dic['displeasure'] = disp

    file2 = open('pickles/featurewords.pkl','wb')
    pickle.dump(dic, file2)


