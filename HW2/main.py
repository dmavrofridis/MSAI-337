import dataloader
import time
import torch.optim as optim
import torch.nn as nn

from global_variables import information, WINDOW, BATCH_SIZE, EMBEDDING_SIZE, LR, EPOCHS
import preprocessing
import FeedForwardText

def main():
    # start = time.time()
    # preprocessing.setup_nltk()
    # text = preprocessing.load_text('wiki.train.txt')
    # text = preprocessing.string_to_lower(text)
    # text = preprocessing.splitting_tokens(text)
    # text = preprocessing.lists_to_tokens(text)
    # text = preprocessing.remove_stopwords(text)
    # text = preprocessing.to_number(text)
    # unique_n = dataloader.unique_words(text)
    # print('unique_words----->' + str( unique_n))
    #
    # mapping = dataloader.create_integers(text)
    # print(type(mapping))
    # reverse_mapping = {i:k for k,i in mapping.items()}
    # integers_texts = dataloader.words_to_integers(text, mapping)# This is X
    # sliced_integers = dataloader.sliding_window(integers_texts, 5)
    # sliced_integers = sliced_integers[:-1] #you cannot predict the final one
    #
    # labels = dataloader.label_generation(integers_texts)
    # print('length_of_input_feature----->' + str( len(sliced_integers)))
    # print('length_of_target_variables----->' + str( len(labels )))
    #
    # # one_hot_dic = dataloader.create_one_hot_encoddings(text, unique_n)
    # # labels_to_vectors = dataloader.integers_to_vectors(labels, reverse_mapping, one_hot_dic)# This is Y
    # # #dataset =  dataloader.wikiDatasetBagOfWords( slised_integers, labels_to_vectors, reverse_mapping, one_hot_dic)
    # # dataset =  dataloader.wikiDataset( slised_integers, labels_to_vectors)
    # print(sliced_integers)
    # dataset = dataloader.TextDataset(integers_texts, labels, window=5)
    # print(dataset.__getitem__(6))
    # dataset_with_batch = dataloader.batch_divder(dataset, batch_size=20)
    # net = FeedForwardNetwork.FeedForward(window_size=5, number_of_classes= unique_n, embedding_space=10)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=0.005)
    # FeedForwardNetwork.train( dataset_with_batch, net,optimizer , criterion, 20 )

    # print('number_of_all_words ------> ' + str(dataloader.overall_words(text)))
    # print('number_of_unique_words ---> '+ str(dataloader.unique_words(text)))
    # one_hot_representation = dataloader.map_words_to_vec(text, one_hot_dic)

    # vectorized_tensors = [torch.LongTensor(vector) for vector in one_hot_representation]

    # vectorized_tensors = dataloader.wikiDataset(one_hot_representation)
    # print('shape_of_1_sample_data_is---------->' + str( vectorized_tensors[0].shape))
    # print('type_of_sample_data_is---------->' + str(type(vectorized_tensors)))
    # loadedData = dataloader.batch_divder(vectorized_tensors, batch_size=20)

    start = time.time()
    preprocessing.setup_nltk()
    text = preprocessing.load_text('wiki.train.txt')
    text = preprocessing.string_to_lower(text)
    text = preprocessing.splitting_tokens(text)
    text = preprocessing.lists_from_tokens(text)
    text = preprocessing.convert_numbers(text)


    unique_tokens = dataloader.unique_words(text)

    if information:
        print("Unique tokens\t\t", str(unique_tokens))
    mapping = dataloader.create_integers(text)
    integertext = dataloader.words_to_integers(text, mapping) #good lord is this stuff painful to use, probably will alter later...
    labels = dataloader.label_generation(integertext)
    reverse_mapping = {i: k for k, i in mapping.items()}  #what is this shit?
    one_hot_dic = dataloader.create_one_hot_encoddings(text, unique_tokens)
    vec_labels = dataloader.integers_to_vectors(labels, reverse_mapping, one_hot_dic)
    dataset = dataloader.TextDataset(integertext, vec_labels, window=WINDOW)

    batched_dataset = dataloader.batch_divder(dataset, batch_size=BATCH_SIZE)

    dat_dere_net = FeedForwardText.FeedForwardText(unique_tokens, EMBEDDING_SIZE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dat_dere_net.parameters(), lr=LR)
    FeedForwardText.train(batched_dataset,dat_dere_net, optimizer, loss_fn, num_epochs=EPOCHS)







    end = time.time() - start
    print(end)


















if __name__ =='__main__':
    main()
