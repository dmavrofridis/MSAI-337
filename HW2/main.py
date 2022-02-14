import preprocessing_fnn
import torch.optim as optim
import torch.nn as nn
import preprocessing
import dataloader
import time
import preprocessing_lstm
def main(lstm = True) :# if false then run FeedFoward
    if lstm == False:
        train_dataset = preprocessing_fnn.preprocess_train_data_FNN(name = 'wiki.train.txt')
        val_dataset =  preprocessing_fnn.preprocess_val_train_data(name = 'wiki.valid.txt')
        preprocessing_fnn.run_feed_forward(train_dataset,val_dataset)
    else:
        train_dataset = preprocessing_lstm.preprocess_train(name = 'wiki.train.txt')
        valid_dataset = preprocessing_lstm.preproces_valid_test(name='wiki.valid.txt')
        preprocessing_lstm.run_lstm(train_dataset,valid_dataset)







if __name__ =='__main__':
    main(lstm =True)