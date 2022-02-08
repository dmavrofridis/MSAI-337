import preprocessing
import dataloader
import time




def main():
    start = time.time()
    preprocessing.setup_nltk()
    text = preprocessing.load_text('wiki.train.txt')
    text = preprocessing.string_to_lower(text)
    text = preprocessing.splitting_tokens(text)
    text = preprocessing.lists_to_tokens(text)
    text = preprocessing.remove_stopwords(text)
    text = preprocessing.to_number(text)
    unique_n = dataloader.unique_words(text)
    print('number_of_all_words ------> ' + str(dataloader.overall_words(text)))
    print('number_of_unique_words ---> '+ str(dataloader.unique_words(text)))
    one_hot_dic = dataloader.create_one_hot_encoddings(text, unique_n)
    one_hot_representation = dataloader.map_words_to_vec(text, one_hot_dic)

    #vectorized_tensors = [torch.LongTensor(vector) for vector in one_hot_representation]


    vectorized_tensors = dataloader.wikiDataset(one_hot_representation)
    print('shape_of_1_sample_data_is---------->' + str( vectorized_tensors[0].shape))
    print('type_of_sample_data_is---------->' + str(type(vectorized_tensors)))
    loadedData = dataloader.batch_divder(vectorized_tensors, batch_size=20)




    end = time.time() - start
    print(end)


















if __name__ =='__main__':
    main()
