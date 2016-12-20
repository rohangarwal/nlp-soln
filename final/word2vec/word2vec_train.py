import gensim, logging, os,pickle
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MySentences(object):
    def __init__(self, filename):
        self.datapkl = pickle.load(open(filename,'rb'))
 
    def __iter__(self):
    	for line in self.datapkl:
    		yield line[0]

if __name__ == "__main__":
	
	
	#sentences = MySentences(filename) # a memory-friendly iterator
	filename = '../dataset/pickles/train_lines.pkl' #train word2vec
	sentences = list()
	datapkl = pickle.load(open(filename,'rb'))
	for i in datapkl:
		sentences.append(i[0].split())
		#sentences.append([i[0].strip()])
		#sentences.append(i[0].decode('utf-8'))

	filename = '../dataset/pickles/test_lines.pkl' #train word2vec
	datapkl = pickle.load(open(filename,'rb'))
	for i in datapkl:
		sentences.append(i[0].split())
		#sentences.append([i[0].strip()])
	
	#train word2vec on test sentences also - check performance.
	#print sentences
	model = gensim.models.Word2Vec(sentences,min_count=1,size=50)

	model.save('w2vmodel')
