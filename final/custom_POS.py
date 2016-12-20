import json
import nltk
from nltk.corpus import stopwords
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
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
    pattern = '''sentiment : {<RB.*>|<JJ.*>}'''

    chunk_rule = nltk.RegexpParser(pattern)
    tree = chunk_rule.parse(t)

    return tree

def Treeparse(tree):
    '''
        To parse chunk tree
    '''
    foods = []
    for subtree in tree.subtrees():
        if subtree.label() == 'sentiment':
            foods.append(' '.join([str(child[0]) for child in subtree]))

    return foods
