import nltk
import re
from nltk.stem.wordnet import WordNetLemmatizer
import sys
sys.stdout.softspace = False


# Happy and sad emoticons

HAPPY = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3', ': ',
    ])

SAD = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])

def lemmatize(line):
    return " ".join([wnl.lemmatize(i) for i in line.split()])

def replace_tweet_elements(line) :
    line = re.sub('#[\w]+','',line)
    line = re.sub('@[\w]+','',line)
    line = line.replace('rt ','')
    line = line.replace('&amp;','and')
    line = line.replace('&gt;','>')
    line = line.replace('&lt;','<')

    return line

def replace_emoticons(line)    :
    for i in HAPPY :
        if i in line :
            line = line.replace(i,'')

    for i in SAD :
        if i in line :
            line = line.replace(i,'')

    return line

def link_remove(line):
    return ' '.join(re.sub("(\w+:\/\/\S+)"," ",line).split())

def seperate(line):
    return " ".join([i for i in re.findall(r"[A-Za-z@#0-9<>]+|\S", line)])

if __name__ == '__main__':
    with open('tweets.txt') as file:
        for line in file:

            #Lowercase
            line = line.lower()

            #Link Removal
            line = link_remove(line)

            #Remove Tweet elements
            line = replace_tweet_elements(line)

            #Remove Emoticons
            line = replace_emoticons(line)

            #Seperation
            line = seperate(line)

            #Lemmatization
            #wnl = WordNetLemmatizer()
            #line = lemmatize(line)

            #Trim
            line = line.strip()

            if(line != ""):
                print line