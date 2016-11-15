import json
import string
from pprint import pprint
from nltk.corpus import stopwords

with open('../neg_amazon_cell_phone_reviews.json') as data_file:    
    data = json.load(data_file)
    
documents = [x['text']+" "+x['summary']+" "+x['summary'] for x in list(data['root'])]
stoplist = set(stopwords.words('english'))
texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]

exclude = set(string.punctuation)
final = []
for line in texts:
    line = [string for string in line]
    final_line = []
    for string in line:
        final_line.append(''.join(ch for ch in string if ch not in exclude))
        
    final_line = ' '.join(i for i in final_line if not i.isdigit() and len(i) > 1)
    flist = final_line.split(" ")[:32]
    final.append(" ".join(flist))
    
with open("neg_sentences.txt", "w") as f:
    f.write("\n".join(final))
