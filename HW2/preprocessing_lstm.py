from preprocessing import *
import dataloader
import FeedForwardNetwork
import torch.optim as optim
import torch.nn as nn
import numpy as np
import LSTM


def preprocess_train(name='wiki.train.txt'):
    setup_nltk()
    text = to_number(lists_to_tokens(splitting_tokens(string_to_lower(load_text(name)))))
    unique_n = dataloader.unique_words(text)
    print('unique_words----->' + str(unique_n))
    mapping = dataloader.create_integers(text)
    reverse_mapping = {i: k for k, i in mapping.items()}
    integers_texts = dataloader.words_to_integers(text, mapping)
    sliced_ints = dataloader.sliding_window(integers_texts, 30)
    sliced_ints = sliced_ints[:-1]
    one_hot_dic = dataloader.create_one_hot_encoddings(text, unique_n)
    labels = dataloader.label_generation_RNN(integers_texts, 30)
    labels_to_vectors = dataloader.integers_to_vectors(labels, reverse_mapping, one_hot_dic)
    dataset = dataloader.wikiDataset(sliced_ints, labels_to_vectors)
    dataset_with_batch = dataloader.batch_divder(dataset, batch_size=20)
    return dataset_with_batch

    # net = FeedForwardNetwork.FeedForward(input_size=30, number_of_classes=27597, embedding_space=100, window_size=30)


def preproccess_valid_test(name='wiki.valid.txt'):
    setup_nltk()
    text = to_number(lists_to_tokens(splitting_tokens(string_to_lower(load_text('wiki.train.txt')))))
    unique_n = dataloader.unique_words(text)
    print('unique_words----->' + str(unique_n))
    mapping = dataloader.create_integers(text)
    reverse_mapping = {i: k for k, i in mapping.items()}
    integers_texts = dataloader.words_to_integers(text, mapping)
    sliced_ints = dataloader.sliding_window(integers_texts, 30)
    sliced_ints = sliced_ints[:-1]
    one_hot_dic = dataloader.create_one_hot_encoddings(text, unique_n)
    validation = to_number(lists_to_tokens(splitting_tokens(string_to_lower(load_text(name)))))
    validation = dataloader.words_to_integers(validation, mapping)
    validation_labels = dataloader.label_generation_RNN(validation, 30)
    validation_sliced_ints = dataloader.sliding_window(validation, 30)
    validation_sliced_ints = validation_sliced_ints[:-1]
    validation_labels_to_vectors = dataloader.integers_to_vectors(validation_labels, reverse_mapping, one_hot_dic)
    val_dataset = dataloader.wikiDataset(validation_sliced_ints, validation_labels_to_vectors)
    val_dataset = dataloader.batch_divder(val_dataset, batch_size=20)
    return val_dataset


def run_lstm(train_dataset, valid_dataset):
    lstm = LSTM.Module()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm.parameters(), lr=0.001)
    LSTM.train(lstm, train_dataset, optimizer, criterion, valid_dataset)
