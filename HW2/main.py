from preprocessing_fnn import *
from preprocessing_lstm import *
import time


def main(lstm=True):  # if false then run FeedFoward

    print("Starting the timer")
    start_time = time.time()

    if not lstm:
        print("Training on a simple Feed Forward Neural Network Model")
        train_dataset = preprocess_train_data_FNN(name='wiki.train.txt')
        val_dataset = preprocess_val_train_data(name='wiki.valid.txt')
        run_feed_forward(train_dataset, val_dataset)
    else:
        print("Training on an LSTM Neural Network Model")
        train_dataset = preprocess_train(name='wiki.train.txt')
        valid_dataset = preproccess_valid_test(name='wiki.valid.txt')
        run_lstm(train_dataset, valid_dataset)

    end_time = time.time() - start_time
    print("Time to completion -> " + str(end_time / 60) + " minutes.")


if __name__ == '__main__':
    main(lstm=False)
