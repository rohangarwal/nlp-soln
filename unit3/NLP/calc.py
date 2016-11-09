import pickle

def accuracy(TP,TN,FP,FN):
    return (TP+TN+0.0) / (TP+TN+FP+FN)

def recall(TP,FN):
    return (TP+0.0) / (TP+FN)

def precision(TP,FP):
    return (TN+0.0) / (TP+FP)

if __name__ == "__main__":

    values = list()
    with open('test_values.pkl', 'rb') as f:
        values = pickle.load(f)

    TP = values[0]
    FN = values[1]
    TN = values[2]
    FP = values[3]
    print "TP = " + str(TP) + "FN = " + str(FN) + "TN = " + str(TN) + "FP = " + str(FP)

    print accuracy(TP,TN,FP,FN)
    print recall(TP,FN)
    print precision(TP,FP)
