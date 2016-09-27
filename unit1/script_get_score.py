import pickle

def get_score(mapping, count, selected_words):
    D = 0.0
    Z = 0.0
    for w1, w2 in selected_words:
        l1 = mapping[w1]
        l2 = mapping[w2]
        union = set(l1 + l2)
        for obj in union:
            t1 = (obj[0], w1, obj[1])
            t2 = (obj[0], w2, obj[1])
            tmp = 0.0
            if t1 in count:
                tmp = count[t1]
                Z += tmp
            if t2 in count:
                tmp -= count[t2]
                Z += count[t2]
                
            D += abs(tmp)
        
    return (1 - D/Z)

if __name__ == "__main__":

	mapping_pkl = open('mapping.pkl', 'rb')
	mapping = pickle.load(mapping_pkl)
	count_pkl = open('count.pkl', 'rb')
	count = pickle.load(count_pkl)
	mapping_pkl.close()
	count_pkl.close()
	selected_words = [('i', 'you'),("taliban", "terrorism"),
	("modi", "modi"),('rahul', 'sonia'),('india', 'uae'),
	('modi', 'pm'),('namo', 'modi'),('muslim', 'arab'),
	('dubai', 'city'),('congress', 'bjp')]
	#selected_words = [('govt', 'government'),]
	score = get_score(mapping, count, selected_words)
	print(score)
