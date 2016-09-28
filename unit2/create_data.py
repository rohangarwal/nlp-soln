import random
from random import randint
tmp = []
for i in range(10000):
    size = randint(2,10)
    ls = random.sample(xrange(0,31), size)
    ls = list(map(str,ls))
    tmp.append(" ".join(ls))
    
for i in range(len(tmp)-1):
    print(tmp[i])
    
print(tmp[-1])
