import pickle
with open('data/0','r') as data:
    l0 = data.readlines()
with open('data/1','r') as data:
    l1 = data.readlines()
with open('data/2','r') as data:
    l2 = data.readlines()
with open('data/3','r') as data:
    l3 = data.readlines()
with open('data/4','r') as data:
    l4 = data.readlines()

dic = {}
dic['f0'] = set([x.strip() for x in l0])
dic['f1'] = set([x.strip() for x in l1])
dic['f2'] = set([x.strip() for x in l2])
dic['f3'] = set([x.strip() for x in l3])
dic['f4'] = set([x.strip() for x in l4])

file = open('pickles/feature.pkl','wb')
pickle.dump(dic,file)
