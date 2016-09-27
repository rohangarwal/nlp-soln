import pickle

def get_vocabulary(corpus):
    return [w for w in set(corpus)]
    
def get_unigram(corpus):
    unigram = dict()
    for w in corpus:
        if w in unigram:
            unigram[w] += 1
        else:
            unigram[w] = 1
            
        
    length_of_corpus = len(corpus)    
    for k, v in sorted(unigram.items(), key=lambda x : x[1], reverse=True):
        unigram[k] = float("{:.5f}".format(float(v)/length_of_corpus))
        
    return unigram
    
def write_new_corpus(words, remove):
    tmp = []
    for w in words:
        if not w in remove:
            tmp.append(w)
        else:
            tmp.append("##token##")
            
    return tmp
    #data = " ".join(tmp)
    #with open("modified_corpus.txt", "w") as f:
        #f.write(data)
        
        
def get_least_occuring_words(unigram):
    tmp = []
    curr = 0.0
    prev = 0.0
    accumulate = 0.0
    for k, curr in sorted(unigram.items(), key=lambda x : x[1]):
        if not prev:
            prev = curr
        accumulate += curr
        if(prev != curr and accumulate >= 0.15):
            break
        prev = curr
        tmp.append(k)
        
    return tmp
    
def create_triplets(words):
    mapping = dict()
    count = dict()
    for i in range(len(words)-2):
        f, s, t = words[i], words[i+1], words[i+2]
        key = (f,s,t)
        if key in count:
            count[key] += 1
        else:
            count[key] = 1
            
        key = s
        if key in mapping:
            mapping[key].append((f,t),)
        else:
            mapping[key] = [(f,t)]
            
    return [mapping, count]
    
if __name__ == "__main__":
    words = open("corpus.txt").read().strip().split(" ")
    #words = open("sample.txt").read().strip().split(" ")
    unigram = get_unigram(words)
    remove = get_least_occuring_words(unigram)
    #print(remove)
    modified_words = write_new_corpus(words, remove)
    #modified_words = open("modified_corpus.txt").read().strip().split(" ")
    mapping, count = create_triplets(modified_words)
    output1 = open('mapping.pkl','wb')
    pickle.dump(mapping,output1)
    output1.close()
    
    output2 = open('count.pkl','wb')
    pickle.dump(count,output2)
    output2.close()
