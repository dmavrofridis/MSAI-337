import preprocessing
import dataloader
import time
import FeedForwardNetwork
import torch.optim as optim
import torch.nn as nn

def main():
    start = time.time()
    text = preprocessing.load_text('wiki.train.txt')
    text = preprocessing.string_to_lower(text)
    text =  preprocessing.splitting_tokens(text)
    text = preprocessing.lists_to_tokens(text)
    text = preprocessing.remove_stopwords(text)
    text = preprocessing.to_number(text)
    unique_n = dataloader.unique_words(text)
    print('unique_words----->' + str( unique_n))

    mapping = dataloader.create_integers(text)
    reverse_mapping = {i:k for k,i in mapping.items()}
    integers_texts = dataloader.words_to_integers(text, mapping)# This is X
    slised_integers = dataloader.sliding_window(integers_texts, 5)
    slised_integers = slised_integers[:-1] #you cannot predict the final one

    labels = dataloader.label_generation(integers_texts)
    print('length_of_input_feature----->' + str( len(slised_integers)))
    print('length_of_target_variables----->' + str( len(labels )))

    one_hot_dic = dataloader.create_one_hot_encoddings(text, unique_n)
    labels_to_vectors = dataloader.integers_to_vectors(labels, reverse_mapping, one_hot_dic)# This is Y
    dataset =  dataloader.wikiDataset( slised_integers, labels_to_vectors)
    dataset_with_batch = dataloader.batch_divder(dataset, batch_size=20)
    net = FeedForwardNetwork.FeedForward(input_size=5, number_of_classes= 27422, embedding_space=100)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    FeedForwardNetwork.train( dataset_with_batch, net,optimizer , criterion, 20 )















    #print('number_of_all_words ------> ' + str(dataloader.overall_words(text)))
   # print('number_of_unique_words ---> '+ str(dataloader.unique_words(text)))
   # one_hot_representation = dataloader.map_words_to_vec(text, one_hot_dic)

    #vectorized_tensors = [torch.LongTensor(vector) for vector in one_hot_representation]


    #vectorized_tensors = dataloader.wikiDataset(one_hot_representation)
    #print('shape_of_1_smaple_data_is---------->' + str( vectorized_tensors[0].shape))
 #   print('type_of_smaple_data_is---------->' + str(type(vectorized_tensors)))


    #loadedData = dataloader.batch_divder(vectorized_tensors, batch_size=20)
    #lis = []








    end = time.time() - start
    print(end)


















if __name__ =='__main__':
    main()
