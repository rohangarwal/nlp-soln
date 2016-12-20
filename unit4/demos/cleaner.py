import pickle 
feature_vec = pickle.load(open("../pickles/featurewords.pkl" , "rb"))
fc = open('comp.txt', 'wa')
fd = open('disp.txt' , 'wa')
fm = open('misc.txt' , 'wa')
for i,j in feature_vec.items():
	if i == "compliment":
		for x in j:
			fc.write(x + "\n")
	elif i == "displeasure"    :
		for x in j:
			fd.write(x + "\n")
	else :
		for x in j:
			fm.write(x+ "\n")