from preprocessing import *
import time


def main(lstm=True):  # if false then run FeedFoward

    print("Starting the timer")
    start_time = time.time()

    if not lstm:
        print("Training on a simple Feed Forward Neural Network Model")
        train_dataset = pre_process_train_data(name='wiki.train.txt', is_LSTM=False)
        valid_dataset = pre_process_val_train_data(name='wiki.valid.txt', is_LSTM=False)
        run_nn_model(train_dataset, valid_dataset, is_LSTM=False)
    else:
        print("Training on an LSTM Neural Network Model")
        train_dataset = pre_process_train_data(name='wiki.train.txt', is_LSTM=True)
        valid_dataset = pre_process_val_train_data(name='wiki.valid.txt', is_LSTM=True)
        run_nn_model(train_dataset, valid_dataset, is_LSTM=True)

    end_time = time.time() - start_time
    print("Time to completion -> " + str(end_time / 60) + " minutes.")


if __name__ == '__main__':
    main(lstm=False)
