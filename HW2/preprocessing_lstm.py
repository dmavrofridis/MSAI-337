import preprocessing
import dataloader
import time
import FeedForwardNetwork
import torch.optim as optim
import torch.nn as nn
import numpy as np
import LSTM

def preprocess():
    start = time.time()
    preprocessing.setup_nltk()
    text = preprocessing.load_text('wiki.train.txt')
    text = preprocessing.string_to_lower(text)
    text = preprocessing.splitting_tokens(text)
    text = preprocessing.lists_to_tokens(text)
    #text = preprocessing.remove_stopwords(text)
    text = preprocessing.to_number(text)
    unique_n = dataloader.unique_words(text)
    print('unique_words----->' + str( unique_n))
    criterion = nn.CrossEntropyLoss()

    mapping = dataloader.create_integers(text)
    reverse_mapping = {i:k for k,i in mapping.items()}
    integers_texts = dataloader.words_to_integers(text, mapping)
    slised_integers = dataloader.sliding_window(integers_texts, 30)
    slised_integers = slised_integers[:-1]
    one_hot_dic = dataloader.create_one_hot_encoddings(text, unique_n)

    labels = dataloader.label_generation_RNN(integers_texts, 30)
    labels_to_vectors = dataloader.integers_to_vectors(labels, reverse_mapping, one_hot_dic)
    dataset =  dataloader.wikiDataset(slised_integers, labels_to_vectors)
    dataset_with_batch = dataloader.batch_divder(dataset, batch_size=20)
    net = FeedForwardNetwork.FeedForward(input_size=30, number_of_classes= 27597, embedding_space=100, window_size=30)
    lstm = LSTM.Module()
    optimizer = optim.Adam(lstm.parameters(), lr=0.001)




    #net = FeedForwardNetwork.FeedForwardText(vocab_size=27597, embedding_size=100)


    # preprocess_validatio_and_test
    validation = preprocessing.load_text('wiki.valid.txt')
    validation = preprocessing.string_to_lower(validation)

    validation = preprocessing.splitting_tokens(validation)
    validation = preprocessing.lists_to_tokens(validation)

    validation = preprocessing.to_number(validation)
    validation = dataloader.words_to_integers(validation, mapping)
    validatiion_labels = dataloader.label_generation_RNN(validation, 30)
    #print('validation_labels_length-------> ' + str(len(validatiion_labels)))
    validatiion_slised_integers = dataloader.sliding_window(validation, 30)
    validatiions_slised_integers = validatiion_slised_integers[:-1]
    validation_labels_to_vectors = dataloader.integers_to_vectors(validatiion_labels, reverse_mapping, one_hot_dic)
    val_dataset = dataloader.wikiDataset(validatiion_slised_integers, validation_labels_to_vectors)
    val_datasett = dataloader.batch_divder(val_dataset, batch_size=20)
    #train_feedforward_network
    #FeedForwardNetwork.train(dataset_with_batch, net, optimizer, criterion,  val_datasett, 2)
    end = time.time() - start
    LSTM.train(lstm, dataset_with_batch, optimizer, criterion,  val_datasett)

    print(end)



if __name__ =='__main__':
    main()