import pickle
with open('pickles/comp.txt','r') as data:
    file1 = data.readlines()
with open('pickles/disp.txt','r') as data:
    file2 = data.readlines()
with open('pickles/misc.txt','r') as data:
    file3 = data.readlines()

dic = {}
dic['compliment'] = set([x.strip() for x in file1])
dic['displeasure'] = set([x.strip() for x in file2])
dic['miscellaneous'] = set([x.strip() for x in file3])

file = open('pickles/featurewords2.pkl','wb')
pickle.dump(dic,file)