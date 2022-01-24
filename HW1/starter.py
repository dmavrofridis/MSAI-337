import nltk
import re
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import numpy as np
#nltk.download()


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

#training_data, testing_data, validation_data = question_3(text)
#QUESTION 4 FAMILY STARTS
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
    count = 0
    for token in set(corpus):
        count +=1
    return count




def question_4_ii(training_corpus):
    count = 0
    for token in (training_corpus):
        count +=1
    return count



def word_to_unk(training_corpus, frequency = 3):
    word_frequncy = {}

    for token in training_corpus:
        if token in word_frequncy:
            word_frequncy[token] +=1
        else:
            word_frequncy[token] = 1
    new_training_corpus=[]
    for token in training_corpus:
        if word_frequncy[token] <=frequency:
            new_training_corpus.append('<unk>')
        else:
            new_training_corpus.append(token)

    return new_training_corpus

def question_iii(new_training_corpus):
    return sum(([1 for i in new_training_corpus if i =='<unk>']))



def question_iv(validation_data, training_data):
    count =0
    for token in validation_data:
        if token not in training_data:
            count +=1
            print(count)
    return count


def question_v(training_data, new_training_data):

    checked =[]
    count = 0
    for index in range(len(training_data)):
        if new_training_data[index] =='<unk>' and training_data[index] not in checked:
            checked.append(training_data[index])
            count +=1
    return count



def question_vi(training_data,stopwords):
    count = 0
    for token in training_data:
        if token in stopwords:
            count +=1
    return count




# QUESTION 4 FAMILY ENDS




























def main():
    stopwords = set_up()

    text = read_file("source_text.txt")
    corpus = my_corpus(None)
    # text = input('Please enter a test sequence to encode and recover: ')

    # Question 1
    text = question_1(text)
    # Question 2
    text = question_2(text)
    # Question 3
    training_data, testing_data, validation_data = question_3(text)
    training_data_without_stop_words = [token for token in training_data if token not in stopwords]
    validation_data_data_without_stop_words = [token for token in validation_data if token not in stopwords]
    testing_data_data_without_stop_words = [token for token in testing_data if token not in stopwords]

    # Question 4
    print('training_data_token_count' + " " + str(question_4_i(training_data_without_stop_words)))
    print('validation_data_token_count' + " " + str(question_4_i( validation_data_data_without_stop_words)))
    print('test_data_token_count' + " " + str(question_4_i(testing_data_data_without_stop_words)))
    print('vocabruary_size' + " " + str(question_4_ii( training_data_without_stop_words)))

    new_training_data = word_to_unk(training_data_without_stop_words)
    print('number_of_unk' +  "   " + str(question_iii(training_data_without_stop_words)))
    print('out_of_words' +  "   " + str(question_iv(validation_data_data_without_stop_words, new_training_data)))
    print('number_of_types' +  "   " + str(question_v(training_data_without_stop_words, new_training_data)))
    print('number_of_stopwords' +  "   " + str(question_vi(training_data, stopwords)))

# QUESTION 4 ends

    #print(question_4(text, stopwords))


    #print(' ')
    #ints = corpus.encode_as_ints(text)
    #print(' ')
    #print('integer encodeing: ', ints)
    #print(' ')

    #text = corpus.encode_as_text(ints)
    #print(' ')
    #print('this is the encoded text: %s' % text)


if __name__ == "__main__":
    main()
