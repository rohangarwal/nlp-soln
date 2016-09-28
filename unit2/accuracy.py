from __future__ import division
import sys

N = sys.argv[1]
trial = sys.argv[2]

if __name__ == '__main__':
    dt = open('results/therotical_'+N+'.out', 'r').read().strip().split("\n")
    analytical = [x.split() for x in dt]

    dt = open('results/test_'+N+'_'+trial+'.out', 'r').read().strip().split("\n")
    model = [x.split() for x in dt]

    total = 0
    correct = 0

    for i in list(range(len(model))):
        md = model[i]
        ana = analytical[i]
        ana = sorted(ana, key = lambda x : int(x))

        for k in list(range(len(ana))):
            if ana[k] == md[k]:
                correct += 1
        total += len(ana)

    print "ACCURACY - ", round((correct / total)*100,3)


