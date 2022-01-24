import nltk
import re
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFKC, Sequence


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


# QUESTION 4 FAMILY STARTS
def question_4_i(corpus):
    count = 0
    for token in set(corpus):
        count +=1
    return count


def question_4_ii(training_corpus):
    return len(training_corpus)


def question_iii(new_training_corpus):
    return sum(([1 for i in new_training_corpus if i =='<unk>']))


def question_iv(validation_data, training_data):
    count = 0
    for token in validation_data:
        if token not in training_data and token != '<unk>':
            count += 1
    return count


def question_v(new_training_data):
    number = [str(i) for i in range(10)]
    type_token ={'string':0, 'number':0}

    for token in new_training_data:
        if token[-1] in number:
            type_token['number'] +=1
        else:
            type_token['string'] +=1
    return type_token


def question_vi(training_data,stopwords):
    count = 0
    for token in training_data:
        if token in stopwords:
            count += 1
    return count


# most encountered
def custom_metric_1(new_training_data, number_of_top_words = 30):
    word_frequncy = {}

    for token in new_training_data:
        if token in word_frequncy:
            word_frequncy[token] +=1
        else:
            word_frequncy[token] = 1

    lis = sorted(word_frequncy, key=word_frequncy.get, reverse=True)[:number_of_top_words]
    final_dic = {}
    for name in lis:
        final_dic[name] = word_frequncy[name]
    return final_dic


# n_gram_creator
def custom_metric_2(new_training_data, n=3, number_of_top_words = 30):
    ngram_list = []
    for index in range(len(new_training_data)):
        if index >= n:
            st = ''
            for i in reversed(range(n)):
                st += new_training_data[index -i] + ' '
            ngram_list.append(st)
    return custom_metric_1(ngram_list, number_of_top_words)

# QUESTION 4 FAMILY ENDS


def main():
    stopwords = set_up()

    text = read_file("source_text.txt")
    x = [text]
    d = "<end_of_passage>"
    for line in x:
        text = [e + d for e in line.split(d) if e]
    training_set, testing_set, validation_set = question_3(text)
    tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
    normalizer = normalizers.Sequence([NFD(), StripAccents(), NFKC(), Lowercase()])
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = Whitespace()

    # We initialize our trainer, giving him the details about the vocabulary we want to generate
    trainer = WordPieceTrainer(vocab_size=5000, special_tokens=["<unk>", "<end_of_passage>", "<start_of_passage>"])
    tokenizer.train_from_iterator(training_set, trainer=trainer)

    training_all_words = ' '.join(training_set)
    vali_all_words = ' '.join(validation_set)
    test_all_words = ' '.join(testing_set)

    training_encoding = tokenizer.encode(training_all_words)
    vali_encoding = tokenizer.encode(vali_all_words)
    test_encoding = tokenizer.encode(test_all_words)

    print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))
    train_vocab = list(tokenizer.get_vocab().keys())
    # training_data_without_stop_words = [token for token in training_encoding.tokens if token not in stopwords]
    # validation_data_data_without_stop_words = [token for token in vali_encoding.tokens if token not in stopwords]

    print('training_data_token_count' + " " + str(question_4_ii(training_encoding.tokens)))
    print('validation_data_token_count' + " " + str(question_4_ii(vali_encoding.tokens)))
    print('test_data_token_count' + " " + str(question_4_ii(test_encoding.tokens)))

    print('number of unk in validation' + "   " + str(question_iii(vali_encoding.tokens)))
    print('number of unk in test' + "   " + str(question_iii(test_encoding.tokens)))

    print('number of out of vocabulary words in validation' + "   " + str(question_iv(vali_encoding.tokens, training_encoding.tokens)))
    print('number of out of vocabulary words in test' + "   " + str(question_iv(test_encoding.tokens, training_encoding.tokens)))

    print('number of types in train' + "   " + str(question_v(training_encoding.tokens)))
    print('number of types in validation' + "   " + str(question_v(vali_encoding.tokens)))
    print('number of types in test' + "   " + str(question_v(test_encoding.tokens)))

    print('number of types in validation' + "   " + str(question_v(training_encoding.tokens)))
    print('number of stopwords in test' + "   " + str(question_vi(vali_encoding.tokens, stopwords)))
    print('number of stopwords in validation' + "   " + str(question_vi(test_encoding.tokens, stopwords)))

    print('top_words in train' + "   " + str(custom_metric_1(training_encoding.tokens, number_of_top_words=30)))
    print('ngram_words in train'+"  " + str(custom_metric_2(training_encoding.tokens, number_of_top_words=120)))



    
    # all_words = ' '.join(training_set)
    # print(len([e for e in all_words if e not in tokenizer.get_vocab().keys()]))
    # encoding = tokenizer.encode(all_words)

    # print("Encoded string: {}".format(encoding.tokens))

    # tokenizer.decoder = decoders.WordPiece()

    # "welcome to the tokenizers library."
    # print("Decoded string: {}".format(tokenizer.decode(encoding.ids)))

if __name__ == "__main__":
    main()
