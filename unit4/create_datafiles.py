import pickle

if __name__ == '__main__':
    file = open('pickles/train.pkl','rb')
    file1 = open('data/tweets_compliment','w')
    file2 = open('data/tweets_displeasure','w')
    file3 = open('data/tweets_miscellaneous','w')
    st1 = ''
    st2 = ''
    st3 = ''
    lists = pickle.load(file)
    for ele in lists:
        if ele[1] == 'miscellaneous':
            st3 += ele[0]+'\n'
        elif ele[1] == 'displeasure':
            st2 += ele[0]+'\n'
        else:
            st1 += ele[0]+'\n'

    file1.write(st1[:-1])
    file2.write(st2[:-1])
    file3.write(st3[:-1])
    file1.close()
    file2.close()
    file3.close()