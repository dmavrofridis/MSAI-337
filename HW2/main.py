
from preprocessing import *
import time
import matplotlib.pyplot as plt
from new_LSTM_pipeline import *

def main(is_LSTM=False, use_custom_loss=False, use_valid=False, upgraded= False):
    'if upgraded equal to false run the old models, otherwise run new pipeline with upgraded LSTM '
    if upgraded == False:


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
        model = pre_process_train_data_LSTM_upgrade(tester='wiki.test.txt')
      #  pre_process_valid_test_data_LSTM_upgrade(model, 'wiki.valid.txt')





if __name__ == '__main__':
    main(is_LSTM=False, use_custom_loss=False, use_valid=False, upgraded =True)
