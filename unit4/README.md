preprocess.py -> Script to preprocess the input, and Dumping the pickle as [tweet, sentiment] for training and testing.

word2vec -> It contains word2vec_train.py and word2vec_run.py to train the model and create the vectors.

RNN -> (i) -> many2one.py -> Trains the RNN Model.
       (ii) -> test_file.py -> Tests the RNN Model and dumps the pickle. It also outputs the ACCURACY, % of SENTIMENT.

create_datafiles.py -> It creates datafiles for MaxEnt

POS_tag.py -> Text Segmentation and Tagging

MaxEnt/bank_tweets.py - client code for MaxEnt
