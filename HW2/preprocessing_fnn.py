from preprocessing import *
import dataloader
import time
import FeedForwardNetwork
import torch.optim as optim
import torch.nn as nn


def preprocess_train_data_FNN(name='wiki.train.txt'):
    setup_nltk()
    text = to_number(lists_to_tokens(splitting_tokens(string_to_lower(load_text(name)))))
    unique_n = dataloader.unique_words(text)
    print('unique_words----->' + str(unique_n))
    mapping = dataloader.create_integers(text)
    reverse_mapping = {i: k for k, i in mapping.items()}
    integers_texts = dataloader.words_to_integers(text, mapping)
    sliced_ints = dataloader.sliding_window(integers_texts, 5)
    sliced_ints = sliced_ints[:-1]  # you cannot predict the final one
    labels = dataloader.label_generation(integers_texts)
    one_hot_dic = dataloader.create_one_hot_encoddings(text, unique_n)
    labels_to_vectors = dataloader.integers_to_vectors(labels, reverse_mapping, one_hot_dic)
    dataset = dataloader.wikiDataset(sliced_ints, labels_to_vectors)
    dataset_with_batch = dataloader.batch_divder(dataset, batch_size=20)
    return dataset_with_batch


def preprocess_val_train_data(name='wiki.valid.txt'):
    setup_nltk()
    text = to_number(lists_to_tokens(splitting_tokens(string_to_lower(load_text('wiki.train.txt')))))
    unique_n = dataloader.unique_words(text)
    mapping = dataloader.create_integers(text)
    reverse_mapping = {i: k for k, i in mapping.items()}
    integers_texts = dataloader.words_to_integers(text, mapping)
    sliced_ints = dataloader.sliding_window(integers_texts, 5)
    sliced_ints = sliced_ints[:-1]  # you cannot predict the final one
    one_hot_dic = dataloader.create_one_hot_encoddings(text, unique_n)
    validation = to_number(lists_to_tokens(splitting_tokens(string_to_lower(load_text(name)))))
    validation = dataloader.words_to_integers(validation, mapping)
    validation_labels = dataloader.label_generation(validation)
    validation_sliced_ints = dataloader.sliding_window(validation, 5)
    validation_sliced_ints = validation_sliced_ints[:-1]
    validation_labels_to_vectors = dataloader.integers_to_vectors(validation_labels, reverse_mapping, one_hot_dic)
    val_dataset = dataloader.wikiDataset(validation_sliced_ints, validation_labels_to_vectors)
    val_datasett = dataloader.batch_divder(val_dataset, batch_size=20)
    return val_datasett


def run_feed_forward(dataset_with_batch, val_dataset):
    criterion = nn.CrossEntropyLoss()
    net = FeedForwardNetwork.FeedForward(input_size=5, number_of_classes=27597, embedding_space=100)
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    FeedForwardNetwork.train(dataset_with_batch, net, optimizer, criterion, val_dataset, 2)