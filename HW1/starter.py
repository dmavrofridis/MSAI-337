import nltk
import re
from nltk.tokenize import word_tokenize


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

def main():

    set_up()

    # text = read_file("source_text.txt")
    corpus = my_corpus(None)
    text = input('Please enter a test sequence to encode and recover: ')
    text = question_1(text)
    text = question_2(text)
    print(text)
    exit(0)
    # Question 1
    text = question_1(text)
    # Question 2
    text = question_2(text)

    print(' ')
    ints = corpus.encode_as_ints(text)
    print(' ')
    print('integer encodeing: ', ints)
    print(' ')

    text = corpus.encode_as_text(ints)
    print(' ')
    print('this is the encoded text: %s' % text)


if __name__ == "__main__":
    main()
