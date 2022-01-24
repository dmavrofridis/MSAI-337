import string

import nltk
import re
from nltk.tokenize import word_tokenize
from global_variables import *
particles = list(string.punctuation)

from sklearn.model_selection import train_test_split
import numpy as np

from nltk.tokenize import RegexpTokenizer

# nltk.download()


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
        print(text[i])
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


# region Question 4
# training_data, testing_data, validation_data = question_3(text)
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


def question_4_i(corpus):
    return len(set(corpus))


def question_4_ii(training_corpus):
    return len(training_corpus)


def word_to_unk(training_corpus, frequency=5):
    word_frequency = {}
    types_counter = {'string': 0, 'date': 0, 'decimal': 0, 'day': 0, 'integer': 0}

    for token in training_corpus:
        if token in word_frequency:
            word_frequency[token] += 1
        else:
            word_frequency[token] = 1
    new_training_corpus = []

    # Here we access the global variable types_to_unk_counter which will count the types we turn to unk
    for token in training_corpus:

        if word_frequency[token] <= frequency:
            if token in list_of_types:
                types_counter[token] += 1
            else:
                types_counter['string'] += 1
            new_training_corpus.append('<unk>')
        else:
            new_training_corpus.append(token)

    return new_training_corpus, types_counter


def question_iii(new_training_corpus):
    return sum(([1 for i in new_training_corpus if i == '<unk>']))


def question_iv(validation_data, training_data):
    count = 0
    gen = (token for token in validation_data if token not in training_data)
    for token in gen:
        count += 1
    return count


def question_vi(training_data, stopwords):
    count = 0
    gen = (token for token in training_data if token in stopwords)
    for token in gen:
        count += 1
    return count


# most encountered
def custom_metric_1(new_training_data, number_of_top_words=30):
    word_frequency = {}

    for token in new_training_data:
        if token in word_frequency:
            word_frequency[token] += 1
        else:
            word_frequency[token] = 1

    lis = sorted(word_frequency, key=word_frequency.get, reverse=True)[:number_of_top_words]
    final_dic = {}
    for name in lis:
        final_dic[name] = word_frequency[name]
    return final_dic


# n_gram_creator
def custom_metric_2(new_training_data, n=3, number_of_top_words=30):
    ngram_list = []
    for index in range(len(new_training_data)):
        if index >= n:
            st = ''
            for i in reversed(range(n)):
                st += new_training_data[index - i] + ' '
            ngram_list.append(st)
    return custom_metric_1(ngram_list, number_of_top_words)


# endregion


def main():
    stopwords = set_up()
    text = read_file("source_text.txt")
    corpus = my_corpus(None)
    # text = input('Please enter a test sequence to encode and recover: ')

    # Question 1
    text = question_1(text)
    # Question 2
    # text = question_2(text)
    # Question 3
    training_data, testing_data, validation_data = question_3(text)
    training_data_without_stop_words = [token for token in training_data if token not in stopwords]
    validation_data_data_without_stop_words = [token for token in validation_data if token not in stopwords]
    testing_data_data_without_stop_words = [token for token in testing_data if token not in stopwords]

    # Question 4
    print('training_data_token_count' + " " + str(question_4_i(training_data_without_stop_words)))
    print('validation_data_token_count' + " " + str(question_4_i(validation_data_data_without_stop_words)))
    print('test_data_token_count' + " " + str(question_4_i(testing_data_data_without_stop_words)))
    print('vocabruary_size' + " " + str(question_4_ii(training_data_without_stop_words)))

    new_training_data, types_counter = word_to_unk(training_data_without_stop_words)
    print('number_of_unk' + "   " + str(question_iii(training_data_without_stop_words)))
    print('out_of_words' + "   " + str(question_iv(validation_data_data_without_stop_words, new_training_data)))
    print('number_of_types' + "   " + str(types_counter))
    print('number_of_stopwords' + "   " + str(question_vi(training_data, stopwords)))
    print('top_words' + "   " + str(custom_metric_1(new_training_data, number_of_top_words=30)))
    print('ngram_words' + "  " + str(custom_metric_2(new_training_data, number_of_top_words=120)))


# QUESTION 4 ends

# print(question_4(text, stopwords))


# print(' ')
# ints = corpus.encode_as_ints(text)
# print(' ')
# print('integer encodeing: ', ints)
# print(' ')

# text = corpus.encode_as_text(ints)
# print(' ')
# print('this is the encoded text: %s' % text)


if __name__ == "__main__":
    main()
