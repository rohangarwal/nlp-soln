import pytreebank
dataset = pytreebank.load_sst('trees')

train = []
for example in dataset['train']:
    for label,line in example.to_labeled_lines():
        train.append(line+":"+str(label))
    
test = []
for example in dataset['test']:
    for label,line in example.to_labeled_lines():
        test.append(line+":"+str(label))
    
with open("train_phrases.txt", "w") as f:
    f.write('\n'.join(train))

with open("test_phrases.txt", "w") as f:
    f.write('\n'.join(test))
    
print("Dataset Created!")
