import preprocessing_fnn
import torch.optim as optim
import torch.nn as nn
import preprocessing
import dataloader
import time
def main(lstm = True) :# if false then run FeedFoward
    train_dataset = preprocessing_fnn.preprocess_train_data_FNN(name = 'wiki.train.txt')
    val_dataset =  preprocessing_fnn.preprocess_val_train_data(name = 'wiki.valid.txt')
    preprocessing_fnn.run_feed_forward(train_dataset,val_dataset)


if __name__ =='__main__':
    main()