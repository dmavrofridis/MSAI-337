import nltk
import re
from nltk.tokenize import RegexpTokenizer
import string

def setup_nltk():
    nltk.download('stopwords')

tokenizer = RegexpTokenizer(r'\w+')
def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        for line in lines:
            line.replace("\n", " </s> ").split()
    return lines




def string_to_lower(text):
    text = [i.lower() for i in text]
    return text

def splitting_tokens(text):
    corpus = []
    for lines in text:
        splitted_lines = lines.split(' ')
        corpus.append(splitted_lines)
    return corpus

def lists_to_tokens(text):
    tokens =[]
    for line in text:
        for word in line:
            if word != '':
                tokens.append(word)
    return tokens
def to_number(text):
    for i in range(len(text)):
        # print(text[i])
        text[i] = re.sub(r'^([0-9]{4})', '<date>', text[i])
        text[i] = re.sub(r'([0-9]+\.[0-9]+)', '<decimal>', text[i])
        text[i] = re.sub(r'^([0-9]{2})$', '<day>', text[i])
        text[i] = re.sub(r'^([0-9]+[^\.0-9][0-9]+)$', '<other>', text[i])
        text[i] = re.sub(r'[0-9]+', '<integer>', text[i])
    return text



def remove_stopwords(text):
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(string.punctuation)
    output= [i for i in text if i not in stopwords]
    return output





