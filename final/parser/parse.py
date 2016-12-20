from stat_parser import Parser, display_tree
import nltk

def get_sentiment(phrase):
    return "NA"

def printTree(node):
    print "ROOT:\n\tLabel :", node.label(), "Text :", ' '.join(node.leaves()), "\n\tSentiment :", get_sentiment(' '.join(node.leaves()))
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

