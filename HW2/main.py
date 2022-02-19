from preprocessing import *
import time
import matplotlib.pyplot as plt
from new_LSTM_pipeline import *


def main(is_LSTM, use_custom_loss, use_valid, use_upgraded_LSTM):
    'if upgraded equal to false run the old models, otherwise run new pipeline with upgraded LSTM '
    if not use_upgraded_LSTM:

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
    else:
        'pre_process_train_data_LSTM_upgrade function trains the model, the second function tests it on a valid dataset'
        print("Starting the timer")
        start_time = time.time()
        data, model, optimizer = pre_process_train_data_LSTM_upgrade(tester='wiki.test.txt')
        train_LSTM_Upgrade(data, model, optimizer, 100, train=True, epoch_size=5)
        end_time = time.time() - start_time
        print("Trained in -> " + str(end_time / 60) + " minutes.")


if __name__ == '__main__':
    main(False, False, False, True)
