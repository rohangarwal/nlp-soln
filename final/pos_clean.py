from custom_POS import *

if __name__ == "__main__":
    filename = 'train_lines.txt'
    filename2 = 'special_train_lines2.txt'
    fp = open(filename2, "w")

    for line in open(filename,'r').readlines():
        line = line.split(':')
        temp = Postag(line[0])
        temp = Chunking(temp)
        temp = Treeparse(temp)
        if temp:
            st = ' '.join(temp) + ':' + line[1]
            fp.write(st)
    fp.close()