from stat_parser import Parser, display_tree
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'word2vec'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Recurrent_NN'))
import word2vec_run
import vanilla_test
import gru_test
import nltk

def get_sentiment(phrase):
    vectors = word2vec_run.get_vectors(phrase)
    if vectors:
        # return vectors
        return gru_test.sentiment(vectors, os.path.join(os.path.dirname(__file__), '..', 'Recurrent_NN', 'gru3_model.pkl'))
        # return vanilla_test.sentiment(vectors, os.path.join(os.path.dirname(__file__), '..', 'Recurrent_NN', 'vanilla5_model.pkl'))
    else:
        return "Word Vector is not available"

def printTree(node):
    print "ROOT:\n\tLabel :", node.label(), "\n\tText :", ' '.join(node.leaves()), "\n\tSentiment :", get_sentiment(' '.join(node.leaves()))
    getNodes(node)
    print '\n\n'

def getNodes(parent):
    for node in parent:
        if type(node) is nltk.Tree:
            print "Nodes:\n\tLabel :", node.label(), "\n\tText :", ' '.join(node.leaves()), "\n\tSentiment :", get_sentiment(' '.join(node.leaves()))
            getNodes(node)

parser = Parser()
# tree = parser.parse("How can the net amount of entropy of the universe be massively decreased?")


# display_tree(tree)
# tree.draw()

# printTree(tree)
# tree.pretty_print()

while True:
    text = raw_input("Enter the sentence : ")
    tree = parser.parse(text)
    printTree(tree)
    tree.pretty_print()
    print '\n\n'

