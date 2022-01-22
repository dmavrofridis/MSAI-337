import nltk
import re
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import numpy as np


class my_corpus:

    def __init__(self, params):
        super().__init__()

        self.params = params
        print('setting parameters')

    def encode_as_ints(self, sequence):
        int_represent = []

        print('encode this sequence: %s' % sequence)
        print('as a list of integers.')

        print(int_represent)

        return int_represent

    def encode_as_text(self, int_represent):
        text = ''

        print('encode this list', int_represent)
        print('as a text sequence.')

        return text


def set_up():
    nltk.download('punkt')
    stopwords = nltk.corpus.stopwords.words('english')
    return stopwords


def read_file(file_name):
    text = open(file_name).read()
    return text


def question_1(text):
    return [token.lower() for token in word_tokenize(text)]


def question_2(text):
    for i in range(len(text)):
        text[i] = re.sub(r'^([0-9]{4})$', '<date>', text[i])
        text[i] = re.sub(r'([0-9]+\.[0-9]+)', '<decimal>', text[i])
        text[i] = re.sub(r'^([0-9]{2})$', '<day>', text[i])
        text[i] = re.sub(r'^([0-9]+[^\.0-9][0-9]+)$', '<other>', text[i])
        text[i] = re.sub(r'[0-9]+', '<integer>', text[i])
    return text


def question_3(text):
    training_data, testing_data = train_test_split(text, test_size=0.2, random_state=25)
    validation_data, testing_data = train_test_split(testing_data, test_size=0.5, random_state=25)
    return training_data, testing_data, validation_data


def question_4(text, stopwords):
    maximum_frequency = 3.0
    word_frequencies = {}
    for word in text:
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    # Cap to the frequence of 3.0
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequency)


def main():
    stopwords = set_up()

    text = read_file("source_text.txt")
    corpus = my_corpus(None)
    # text = input('Please enter a test sequence to encode and recover: ')
    text = read_file("source_text.txt")

    x = text.split("<<end_of_passage>>")
    d = "<end_of_passage>"
    for line in x:
        text = [e + d for e in line.split(d) if e]
    # Question 3
    # Should be much more convenient if we slpit the set before we tokenize them.
    training_set, testing_set, validation_set = question_3(text)
    print(len(training_set),len(testing_set),len(validation_set))
    # Question 1
    training_set,testing_set,validation_set = question_1(' '.join(training_set)),question_1(
        ' '.join(testing_set)),question_1(' '.join(validation_set))
    # Question 2
    training_set, testing_set, validation_set = question_2(training_set), question_2(testing_set),\
                                                question_2(validation_set)
    print(validation_set)
    # Question 4
    question_4(training_set, stopwords)
    question_4(testing_set, stopwords)
    question_4(validation_set, stopwords)

    #
    # print(' ')
    # ints = corpus.encode_as_ints(text)
    # print(' ')
    # print('integer encodeing: ', ints)
    # print(' ')
    #
    # text = corpus.encode_as_text(ints)
    # print(' ')
    # print('this is the encoded text: %s' % text)


if __name__ == "__main__":
    main()
