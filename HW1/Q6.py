import nltk
import re
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import numpy as np
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE, WordPiece
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
from tokenizers import decoders
from tokenizers.pre_tokenizers import Whitespace


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
    x = [text]
    d = "<end_of_passage>"
    for line in x:
        text = [e + d for e in line.split(d) if e]
    training_set, testing_set, validation_set = question_3(text)
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    normalizer = normalizers.Sequence([NFD(), StripAccents()])
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = Whitespace()

    # We initialize our trainer, giving him the details about the vocabulary we want to generate
    trainer = WordPieceTrainer(vocab_size=5000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train_from_iterator(training_set, trainer=trainer)

    print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))
    print("Trained vocab statistics.: {}".format(tokenizer.get_vocab()))
    
    # all_words = ' '.join(training_set)
    # print(len([e for e in all_words if e not in tokenizer.get_vocab().keys()]))
    # encoding = tokenizer.encode(all_words)

    # print("Encoded string: {}".format(encoding.tokens))

    # tokenizer.decoder = decoders.WordPiece()

    # "welcome to the tokenizers library."
    # print("Decoded string: {}".format(tokenizer.decode(encoding.ids)))

if __name__ == "__main__":
    main()
