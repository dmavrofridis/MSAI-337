from preprocessing import *
import time


def main(is_LSTM=False, use_custom_loss=False):  # if false then run FeedFoward

    if is_LSTM:
        print("Training on an LSTM Neural Network Model")
        dataset = 'wiki.valid.txt'

    else:
        print("Training on a simple Feed Forward Neural Network Model")
        dataset = 'wiki.test.txt'

    train_dataset = pre_process_train_data(name='wiki.train.txt', is_LSTM=is_LSTM)
    valid_dataset = pre_process_val_train_data(name=dataset, is_LSTM=is_LSTM)
    print("Starting the timer")
    start_time = time.time()
    run_nn_model(train_dataset, valid_dataset, is_LSTM=is_LSTM, use_custom_loss=use_custom_loss)
    end_time = time.time() - start_time
    print("Trained in -> " + str(end_time / 60) + " minutes.")


if __name__ == '__main__':
    main(is_LSTM=True, use_custom_loss=True)
