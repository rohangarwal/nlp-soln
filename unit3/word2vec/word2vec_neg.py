import gensim, pickle, random

if __name__ == "__main__":

	model = gensim.models.Word2Vec.load('data/w2vmodel')

	filename = "neg_sentences.txt"
	all_neg_vecs = list()
	with open(filename) as f:
		lines = f.readlines()
		while(len(all_neg_vecs) != 1100):
			line = lines[random.randint(0,len(lines))]
			review_vec = list()
			for word in line.strip().split():
				if word in model.vocab:
					review_vec.append(model[word])
			all_neg_vecs.append(review_vec)

	with open('neg_vec_train.pkl', 'wb') as f:
		pickle.dump(all_neg_vecs[:1000],f)

	with open('neg_vec_test.pkl', 'wb') as f:
		pickle.dump(all_neg_vecs[1000:],f)