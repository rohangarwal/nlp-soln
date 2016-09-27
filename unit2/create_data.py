import random
from random import randint

for i in range(10000):
    size = randint(2,10)
    ls = random.sample(xrange(0,31), size)
    ls = list(map(str,ls))
    print " ".join(ls)
