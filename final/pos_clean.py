from custom_POS import *
import sys

if __name__ == "__main__":
    filename = sys.argv[1]
    filename2 = sys.argv[2]
    fp = open(filename2, "w")
    lines_seen = set()
    for line in open(filename,'r').readlines():
        line = line.split(':')
        temp = Postag(line[0])
        
        if temp:
        	temp = Chunking(temp)
        	if temp:
        	    temp = Treeparse(temp)
        	    mn = ' '.join(temp)
        	    if mn not in lines_seen:
        	        lines_seen.add(mn)
        	        st = mn + ':' + line[1]
        	        fp.write(st)
    fp.close()
