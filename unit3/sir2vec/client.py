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

serviceURL = "http://www.jnresearchlabs.com:9027/" # NOTE: we will get rid of port number later!
runCommandURL = serviceURL + "run_command" # this is the end point to which we will POST the command and params
headers = {'content-type': 'application/json'}

def get_vec_for_words(words): 
    """Given a list of words return the corresponding word vectors - default dimensions = 32"""
    r = requests.post(runCommandURL, data = json.dumps({"cmd": "word_reps", "params": {"txt": words}}), headers = headers) #
    return json.loads(r.text)

def get_vec_for_words_2d(words): 
    """Given a list of words return the corresponding word vectors as 2d vectors. This is useful for visualizations"""
    r = requests.post(runCommandURL, data = json.dumps({"cmd": "word_reps_2d", "params": {"txt": words}}), headers = headers) #
    return json.loads(r.text)

def get_named_entities(txt):
    """Given a text returns the Named Entities - txt is rate limited to < 10K at a time""" 
    r = requests.post(runCommandURL, data = json.dumps({"cmd": "ner", "params": {"txt": txt}}), headers = headers) #
    return json.loads(r.text)

def test_service():
    inp1 = raw_input("Which test you want to run (NER/WR) default is NER: ")
    if inp1 == "__Q__":
        print "Quitting due to quit command"
        return
    while True:
        if inp1 == "WR":
            inp = raw_input("Enter words for getting vectors: ")
            result = get_vec_for_words(inp.split())
            result_2d = get_vec_for_words_2d(inp.split())
            print result
            print result_2d
        else: # inp1 == "NER"
            inp = raw_input("Enter text for getting named entities (rate limited to 10k): ")
            result = get_named_entities(inp)
            print
            print "***** Your service request returned the response code/message: ", result["response_code"], " ", result["response_message"], " ****"
            print "Supported Tags are: ", result["supported_tags"]
            print
            for record in result["tags"]:
                print "-" * 40, " Richer NE tags ", "-" * 40
                for w, t in zip(record["word"], record["tag"]):
                    print w, "\t\t\t=>\t\t", t
                print "-" * 40, " Regular NE tags ", "-" * 40
                for t in record["map"]:
                    print t[0], "\t\t\t=>\t\t", t[1]
        
    return

if __name__ == '__main__':
    test_service()
