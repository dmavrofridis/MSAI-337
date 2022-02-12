import preprocessing
import dataloader
import time
import FeedForwardNetwork
import torch.optim as optim
import torch.nn as nn
import numpy as np
import LSTM

def main():
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

    mapping = dataloader.create_integers(text)
    reverse_mapping = {i:k for k,i in mapping.items()}
    integers_texts = dataloader.words_to_integers(text, mapping)
    slised_integers = dataloader.sliding_window(integers_texts, 5)
    slised_integers = slised_integers[:-1] #you cannot predict the final one

    labels = dataloader.label_generation(integers_texts)
 #   print('length_of_input_feature----->' + str( len(slised_integers)))
   # print('length_of_target_variables----->' + str( len(labels )))

    one_hot_dic = dataloader.create_one_hot_encoddings(text, unique_n)
    labels_to_vectors = dataloader.integers_to_vectors(labels, reverse_mapping, one_hot_dic)
    #dataset =  dataloader.wikiDatasetBagOfWords( slised_integers, labels_to_vectors, reverse_mapping, one_hot_dic)
    dataset =  dataloader.wikiDataset( slised_integers, labels_to_vectors)
    dataset_with_batch = dataloader.batch_divder(dataset, batch_size=20)
    net = FeedForwardNetwork.FeedForward(input_size=5, number_of_classes= 27597, embedding_space=100)
    #net = FeedForwardNetwork.FeedForwardText(vocab_size=27597, embedding_size=100)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    # preprocess_validatio_and_test
    validation = preprocessing.load_text('wiki.valid.txt')
    validation = preprocessing.string_to_lower(validation)

    validation = preprocessing.splitting_tokens(validation)
    validation = preprocessing.lists_to_tokens(validation)

    validation = preprocessing.to_number(validation)
    validation = dataloader.words_to_integers(validation, mapping)
    validatiion_labels = dataloader.label_generation(validation)
    #print('validation_labels_length-------> ' + str(len(validatiion_labels)))
    validatiion_slised_integers = dataloader.sliding_window(validation, 5)
    validatiions_slised_integers = validatiion_slised_integers[:-1]
    validation_labels_to_vectors = dataloader.integers_to_vectors(validatiion_labels, reverse_mapping, one_hot_dic)
    val_dataset = dataloader.wikiDataset(validatiion_slised_integers, validation_labels_to_vectors)
    val_datasett = dataloader.batch_divder(val_dataset, batch_size=20)
    #train_feedforward_network
    FeedForwardNetwork.train(dataset_with_batch, net, optimizer, criterion,  val_datasett, 2)
    # Here wwe preprocess LSTM








    text = preprocessing.load_text('wiki.train.txt')
    text = preprocessing.string_to_lower(text)
    text = preprocessing.splitting_tokens(text)
    text = preprocessing.lists_to_tokens(text)
    #text = preprocessing.remove_stopwords(text)
    text = preprocessing.to_number(text)
    unique_n = dataloader.unique_words(text)
    print('unique_words----->' + str( unique_n))

    mapping = dataloader.create_integers(text)
    reverse_mapping = {i:k for k,i in mapping.items()}
    integers_texts = dataloader.words_to_integers(text, mapping)
    slised_integers = dataloader.sliding_window(integers_texts, 5)
    slised_integers = slised_integers[:-1]
    labels = dataloader.label_generation_RNN(integers_texts, 5)
    labels_to_vectors = dataloader.integers_to_vectors(labels, reverse_mapping, one_hot_dic)
    dataset =  dataloader.wikiDataset(slised_integers, labels_to_vectors)
    dataset_with_batch = dataloader.batch_divder(dataset, batch_size=20)
    net = FeedForwardNetwork.FeedForward(input_size=5, number_of_classes= 27597, embedding_space=100, window_size=5)
    FeedForwardNetwork.train(dataset_with_batch, net, optimizer, criterion,  val_datasett, 2)






























    #print('number_of_all_words ------> ' + str(dataloader.overall_words(text)))
   # print('number_of_unique_words ---> '+ str(dataloader.unique_words(text)))
   # one_hot_representation = dataloader.map_words_to_vec(text, one_hot_dic)

    #vectorized_tensors = [torch.LongTensor(vector) for vector in one_hot_representation]



    end = time.time() - start
    print(end)


















if __name__ =='__main__':
    main()
