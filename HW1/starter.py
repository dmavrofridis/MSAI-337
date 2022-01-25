import string

import nltk
import re
import random
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
    date= {}
    decimal={}
    day ={}
    integer = {}
    other = {}
    for i in range(len(text)):
        if  re.match(r'^([0-9]{4})',text[i] ):
            print(text[i])
            if text[i] not in date:
                date[text[i]] =1
            else:
                date[text[i]] +=1
            text[i] = '<date>'

        elif  re.match(r'([0-9]+\.[0-9]+)',text[i] ):
            print(text[i])
            if text[i] not in decimal:
                decimal[text[i]] =1
            else:
                decimal[text[i]] +=1
            text[i] = '<decimal>'

        elif re.match(r'([0-9]+\.[0-9]+)',text[i] ):
            print(text[i])

            if  text[i] not in day:
                day[text[i]] =1
            else:
                day[text[i]] +=1
            text[i] = '<day>'

        elif re.match(r'([0-9]+\.[0-9]+)',text[i] ):
            print(text[i])

            if text[i] not in other:
                other[text[i]] =1
            else:
                other[text[i]] +=1
            text[i] = '<other>'

        elif re.match(r'([0-9]+\.[0-9]+)',text[i] ):
            print(text[i])

            if text[i] not in integer:
                integer[text[i]] =1
            else:
                integer[text[i]] +=1
            text[i] = '<integer>'
    print(date)
    return text, date, decimal, day, other, integer
'''

def question_2(text):
    for i in range(len(text)):
        # print(text[i])
        text[i] = re.sub(r'^([0-9]{4})', '<date>', text[i])
        text[i] = re.sub(r'([0-9]+\.[0-9]+)', '<decimal>', text[i])
        text[i] = re.sub(r'^([0-9]{2})$', '<day>', text[i])
        text[i] = re.sub(r'^([0-9]+[^\.0-9][0-9]+)$', '<other>', text[i])
        text[i] = re.sub(r'[0-9]+', '<integer>', text[i])
    return text
'''
def question_3(text):
    # training_data, testing_data = train_test_split(text, test_size=0.2, random_state=25)
    # validation_data, testing_data = train_test_split(testing_data, test_size=0.5, random_state=25)

    training_data = []
    testing_data = []
    validation_data = []

    length = len(text)

    order = range(0, length - 1)
    order = list(order)
    random.shuffle(order)

    for i in range(length):
        if i % 10 == 9:
            testing_data.append(text[i])
        elif i % 10 == 1:
            validation_data.append(text[i])
        else:
            training_data.append(text[i])

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

    return new_training_corpus

def types_to_unk(date, decimal, day, other, integer, threshold =5):
    date_c, decimal_c, day_c, other_c, integer_c, threshhold_c = 0,0,0,0,0,0
    for i in date:
        if date[i] <=5:
            date_c +=1
    for i in decimal:
        if decimal[i] <= 5:
            decimal_c += 1

    for i in day:
        if day[i] <=5:
            day_c +=1
    for i in other:
        if other[i] <= 5:
            other_c += 1

    for i in integer:
        if integer[i] <= 5:
            integer += 1

    return date_c, decimal_c, day_c, other_c, integer_c, threshhold_c



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
    particles.extend(['', '==', '', '``', "'s",  "s" ,'===',"''", '.', ',', '(', ')', "''", ])
    stopwords.extend(particles)


    # Question 1
    text = question_1(text)
    # Question 2
    text, date, decimal, day, other, integer = question_2(text)
    print('types_to_ukn' + "   "  + str(types_to_unk( date, decimal, day, other, integer)))
    # Question 3
    training_data, testing_data, validation_data = question_3(text)



    #training_data_without_stop_words = [token for token in training_data if token not in stopwords or token not in particles]
    training_data_without_stop_words = [token for token in training_data if token not in stopwords and token not in particles]

    validation_data_data_without_stop_words = [token for token in validation_data if token not in stopwords and token not in particles]
    testing_data_data_without_stop_words = [token for token in testing_data if token not in stopwords and token not in particles]

    # Question 4
    print('vocabruary size' + " " + str(question_4_i(training_data_without_stop_words)))
    print('validation_data_token_count' + " " + str(question_4_i(validation_data_data_without_stop_words)))
    print('test_data_token_count' + " " + str(question_4_i(testing_data_data_without_stop_words)))
    print('training_data_token_count' + " " + str(question_4_ii(training_data_without_stop_words)))

    new_training_data = word_to_unk(training_data_without_stop_words)
    print('number_of_unk' + "   " + str(question_iii(new_training_data)))
    print('out_of_words' + "   " + str(question_iv(validation_data_data_without_stop_words, new_training_data)))
    #print('number_of_types' + "   " + str(types_counter))
    print('number_of_stopwords' + "   " + str(question_vi(training_data, stopwords)))
    print('top_words' + "   " + str(custom_metric_1(new_training_data, number_of_top_words=100)))
    print('ngram_words' + "  " + str(custom_metric_2(new_training_data, number_of_top_words=100)))


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
