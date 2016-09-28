import random
from random import randint
import sys

N = int(sys.argv[1])
tmp = []
for i in range(N):
    size = randint(2,10)
    ls = random.sample(xrange(0,31), size)
    ls = list(map(str,ls))
    tmp.append(" ".join(ls))
    
ftrain = open("inputs/train_"+str(N)+".in", "w")
ftest = open("inputs/test_"+str(N)+".in", "w")   
ftrain.write("\n".join(tmp[:int(N*0.8)]))
ftest.write("\n".join(tmp[int(N*0.8):]))

ftherotical = open('results/therotical_'+str(N)+'.out', 'w')
s = [sorted(list(map(int, x.split(" ")))) for x in tmp[int(N*0.8):]]
s = '\n'.join([' '.join([str(j) for j in x]) for x in s])
ftherotical.write(s)
