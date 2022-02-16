from preprocessing import *
import time
import matplotlib.pyplot as plt


def main(is_LSTM=False, use_custom_loss=False, use_valid=False):  # if false then run FeedFoward

    dataset = 'wiki.valid.txt' if use_valid else 'wiki.test.txt'
    to_print = "Training on an LSTM Neural Network Model" if is_LSTM else "Training on a simple Feed Forward Neural Network Model"
    print(to_print)

    train_dataset = pre_process_train_data(name='wiki.train.txt', is_LSTM=is_LSTM)
    valid_dataset = pre_process_val_train_data(name=dataset, is_LSTM=is_LSTM)
    print("Starting the timer")
    start_time = time.time()
    run_nn_model(train_dataset, valid_dataset, is_LSTM=is_LSTM, epoch=1, use_custom_loss=use_custom_loss)
    end_time = time.time() - start_time
    print("Trained in -> " + str(end_time / 60) + " minutes.")


if __name__ == '__main__':
    main(is_LSTM=False, use_custom_loss=False, use_valid=False)
