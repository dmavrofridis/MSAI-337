import nltk
import re
from dataloader import *
from nltk.tokenize import RegexpTokenizer
import string
import FeedForwardNetwork
import torch.optim as optim
import torch.nn as nn
import LSTM
from global_variables import *


def pre_process_train_data(name='wiki.train.txt', is_LSTM=False):
    setup_nltk()
    if is_LSTM:
        sliding_window_value = 30
    else:
        sliding_window_value = 5
    text = to_number(lists_to_tokens(splitting_tokens(string_to_lower(load_text(name)))))
    unique_n = unique_words(text)
    print('unique_words----->' + str(unique_n))
    mapping = create_integers(text)
    reverse_mapping = {i: k for k, i in mapping.items()}
    integers_texts = words_to_integers(text, mapping)
    overall_numbers = len(integers_texts)
    sliced_ints = sliding_window(integers_texts, sliding_window_value)
    sliced_ints = sliced_ints[:-1]  # you cannot predict the final one
    one_hot_dic = create_one_hot_encoddings(text, unique_n)
    labels = label_generation_RNN(integers_texts, 30) if is_LSTM else label_generation(integers_texts)
    labels_to_vectors = integers_to_vectors(labels, reverse_mapping, one_hot_dic)
    dataset = wikiDataset(sliced_ints, labels_to_vectors)
    dataset_with_batch = batch_divder(dataset, batch_size=  BATCH_SIZE )

    return dataset_with_batch


def pre_process_val_train_data(name='wiki.valid.txt', is_LSTM=False):
    if is_LSTM:
        sliding_window_value = 30
    else:
        sliding_window_value = 5
    setup_nltk()
    text = to_number(lists_to_tokens(splitting_tokens(string_to_lower(load_text('wiki.train.txt')))))
    #text = remove_stopwords(text)
    unique_n = unique_words(text)
    mapping = create_integers(text)
    reverse_mapping = {i: k for k, i in mapping.items()}
    integers_texts = words_to_integers(text, mapping)
    overall_numbers = len(integers_texts)

    sliced_ints = sliding_window(integers_texts, sliding_window_value)
    sliced_ints = sliced_ints[:-1]  # you cannot predict the final one
    one_hot_dic = create_one_hot_encoddings(text, unique_n)
    validation = to_number(lists_to_tokens(splitting_tokens(string_to_lower(load_text(name)))))
    validation = words_to_integers(validation, mapping)
    validation_labels = label_generation_RNN(validation, 30) if is_LSTM else label_generation(validation)
    validation_sliced_ints = sliding_window(validation, 30) if is_LSTM else sliding_window(validation, 5)
    validation_sliced_ints = validation_sliced_ints[:-1]
    validation_labels_to_vectors = integers_to_vectors(validation_labels, reverse_mapping, one_hot_dic)
    val_dataset = wikiDataset(validation_sliced_ints, validation_labels_to_vectors)
    val_datasett = batch_divder(val_dataset, batch_size=   BATCH_SIZE )
    return val_datasett


def softmax(x):
    exp_x = torch.exp(x)
    sum_x = torch.sum(exp_x, dim=1, keepdim=True)
    return exp_x / sum_x


def log_softmax(x):
    return x - torch.logsumexp(x, dim=1, keepdim=True)


def custom_cross_entropy_loss(outputs, targets):
    num_examples = targets.shape[0]
    batch_size = outputs.shape[0]
    outputs = log_softmax(outputs)
    outputs = outputs[range(batch_size), targets]
    return - torch.sum(outputs) / num_examples


def run_nn_model(train_dataset, val_dataset, is_LSTM=False, epoch=1, use_custom_loss=False):
    if is_LSTM:
        lstm = LSTM.Module()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(lstm.parameters(), lr=0.001)
        LSTM.train(lstm, train_dataset, optimizer, criterion, val_dataset, epoch, use_custom_loss)
    else:
        criterion = nn.CrossEntropyLoss()
        net = FeedForwardNetwork.FeedForward(input_size=5, number_of_classes=27597, embedding_space=100)
        optimizer = optim.Adam(net.parameters(), lr=0.01)
        FeedForwardNetwork.train(net, train_dataset,  optimizer, criterion, val_dataset, epoch =1, use_custom_loss= False)


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
    tokens = []
    for line in text:
        for word in line:
            if word != '':
                tokens.append(word)
    return tokens


def to_number(text):
    for i in range(len(text)):
        text[i] = re.sub(r'^([0-9]{4})', '<date>', text[i])
        text[i] = re.sub(r'([0-9]+\.[0-9]+)', '<decimal>', text[i])
        text[i] = re.sub(r'^([0-9]{2})$', '<day>', text[i])
        text[i] = re.sub(r'^([0-9]+[^\.0-9][0-9]+)$', '<other>', text[i])
        text[i] = re.sub(r'[0-9]+', '<integer>', text[i])
    return text


def remove_stopwords(text):
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(string.punctuation)
    output = [i for i in text if i not in stopwords]
    return output
