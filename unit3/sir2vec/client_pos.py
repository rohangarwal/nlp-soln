'''
Created on 31-Oct-2016
@author: Anantharaman Palacode Narayana Iyer
This is a Python client for the neural_server
Basic Usage: runCommandURL is the end point URL that requires the parameter as POST parameters
You need to pass a dictionary with cmd as the command type and params that contain the parameters
See the example below.

get_vec_for_words(list_of_words) : returns {"cmd": "word_reps", "reps": [w1_rep, ...wn_rep], "words": list_of_words_passed}
w1_rep is the 32 dimensional word vector for word w1 and so on

get_vec_for_words_2d(list_of_words) : returns {"cmd": "word_reps", "reps": [w1_rep, ...wn_rep], "words": list_of_words_passed}
w1_rep is the 2 dimensional word vector for word w1 and so on (same as before but d is 2)
'''
import requests
import json
import random
import pickle

serviceURL = "http://www.jnresearchlabs.com:9027/" # NOTE: we will get rid of port number later!
runCommandURL = serviceURL + "run_command" # this is the end point to which we will POST the command and params
headers = {'content-type': 'application/json'}

def get_vec_for_words(words): 
    """Given a list of words return the corresponding word vectors - default dimensions = 32"""
    r = requests.post(runCommandURL, data = json.dumps({"cmd": "word_reps", "params": {"txt": words}}), headers = headers) #
    return json.loads(r.text)

if __name__ == '__main__':
    filename = "pos_sentences.txt"
    all_neg_vecs = list()
    with open(filename) as f:
        lines = f.readlines()
        while(len(all_neg_vecs) != 1100):
            line = lines[random.randint(0,len(lines))]
            result = get_vec_for_words(line.split())['reps']
            all_neg_vecs.append(result)
            print(len(all_neg_vecs))

	with open('pos_vec_train.pkl', 'wb') as f:
		pickle.dump(all_neg_vecs[:1000],f)

	with open('pos_vec_test.pkl', 'wb') as f:
		pickle.dump(all_neg_vecs[1000:],f)
