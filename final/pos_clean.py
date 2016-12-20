from custom_POS import *
import sys

if __name__ == "__main__":
    filename = sys.argv[1]
    filename2 = sys.argv[2]
    fp = open(filename2, "w")

    for line in open(filename,'r').readlines():
        line = line.split(':')
        temp = Postag(line[0])
        
        if temp:
        	temp = Chunking(temp)
        	temp = Treeparse(temp)
        	if temp:
        		st = ' '.join(temp) + ':' + line[1]
        		fp.write(st)
    fp.close()
