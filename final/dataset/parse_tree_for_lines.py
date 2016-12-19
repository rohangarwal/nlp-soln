import pytreebank
dataset = pytreebank.load_sst('trees')

train = []
for example in dataset['train']:
    train.append(example.to_lines()[0]+":"+str(example.label))
    
test = []
for example in dataset['test']:
    test.append(example.to_lines()[0]+":"+str(example.label))
    
with open("train_lines.txt", "w") as f:
    f.write('\n'.join(train))

with open("test_lines.txt", "w") as f:
    f.write('\n'.join(test))
    
print("Dataset Created!")
