from preprocessing import *
import dataloader
import time
import FeedForwardNetwork
import torch.optim as optim
import torch.nn as nn
import numpy as np


def main():

    start = time.time()
    setup_nltk()
    training_data = to_number(lists_to_tokens(splitting_tokens(string_to_lower(load_text('wiki.train.txt')))))
    # Create integer encodings
    unique_n = dataloader.unique_words(training_data)
    print('unique_words----->' + str(unique_n))
    mapping = dataloader.create_integers(training_data)
    reverse_mapping = {i: k for k, i in mapping.items()}
    # Assign the integer values
    integers_texts = dataloader.words_to_integers(training_data, mapping)  # This is X
    sliced_integers = dataloader.sliding_window(integers_texts, 5)
    sliced_integers = sliced_integers[:-1]  # you cannot predict the final one
    labels = dataloader.label_generation(integers_texts)
    print('length_of_input_feature----->' + str(len(sliced_integers)))
    print('length_of_target_variables----->' + str(len(labels)))
    one_hot_dic = dataloader.create_one_hot_encoddings(training_data, unique_n)
    labels_to_vectors = dataloader.integers_to_vectors(labels, reverse_mapping, one_hot_dic)
    print(labels_to_vectors[0])
    # dataset =  dataloader.wikiDatasetBagOfWords( slised_integers, labels_to_vectors, reverse_mapping, one_hot_dic)
    dataset = dataloader.wikiDataset(sliced_integers, labels_to_vectors)
    dataset_with_batch = dataloader.batch_divder(dataset, batch_size=20)

    # Creating the NET
    net = FeedForwardNetwork.FeedForward(input_size=5, number_of_classes=27597, embedding_space=100)
    # net = FeedForwardNetwork.FeedForwardText(vocab_size=27597, embedding_size=100)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    # preprocess_validation_and_test Sets
    validation_data = to_number(lists_to_tokens(splitting_tokens(string_to_lower(load_text('wiki.valid.txt')))))
    validation_data = dataloader.words_to_integers(validation_data, mapping)
    validation_labels = dataloader.label_generation(validation_data)
    print('validation_labels_length-------> ' + str(len(validation_labels)))
    validation_sliced_integers = dataloader.sliding_window(validation_data, 5)
    validation_sliced_integers = validation_sliced_integers[:-1]
    validation_labels_to_vectors = dataloader.integers_to_vectors(validation_labels, reverse_mapping, one_hot_dic)
    val_dataset = dataloader.wikiDataset(validation_sliced_integers, validation_labels_to_vectors)
    val_datasett = dataloader.batch_divder(val_dataset, batch_size=20)
    # Run the Model
    FeedForwardNetwork.train(dataset_with_batch, net, optimizer, criterion, val_datasett, 20)

    # print('number_of_all_words ------> ' + str(dataloader.overall_words(text)))
    # print('number_of_unique_words ---> '+ str(dataloader.unique_words(text)))
    # one_hot_representation = dataloader.map_words_to_vec(text, one_hot_dic)

    # vectorized_tensors = [torch.LongTensor(vector) for vector in one_hot_representation]

    end = time.time() - start
    print(end)


if __name__ == '__main__':
    main()
