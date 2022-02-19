import torch.nn as nn
import torch
import torch.nn.functional as F
import nltk
from global_variables import *
from preprocessing import *

class LSTM_Language_Model(nn.Module):
    def __init__(self, vocab_size=27597, embedding_dim=100,
                 hidden_dim=100, lstm_layers=2, dropout=0.2):
        super(LSTM_Language_Model, self).__init__()

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, lstm_layers, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, vocab_size)
        self.hl = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.hl = nn.Linear(hidden_dim, hidden_dim)

    def embedd(self, word_indexes):
        return self.fc1.weight.index_select(0, word_indexes)

    def forward(self, packed_sents):
        emb_seq = nn.utils.rnn.PackedSequence(
            self.embedd(packed_sents.data), packed_sents.batch_sizes)
        result, _ = self.lstm(emb_seq)

        hl = self.hl(result.data)
        hl = self.activation(hl)
        hl = self.activation(hl)
        hl = self.activation(hl)

        out = self.fc1(hl)
        return F.log_softmax(out, dim=1)


def divider(data, size=BATCH_SIZE, time=30, window=30):
    batch = []
    count = 0
    for i in range(1, len(data) + 1, window + 1):
        count += 1
        sequence = data[i - 1:i - 1 + time + 1]
        batch.append(sequence)
        if count != 0 and count % size == 0:
            tmp_batch = batch
            tmp_batch.sort(key=lambda l: len(l), reverse=True)
            batch = []
            yield tmp_batch


def pre_process_train_data_LSTM_upgrade(name='wiki.train.txt', is_LSTM=True):
    setup_nltk()
    sliding_window_value = 30
    text = to_number(lists_to_tokens(splitting_tokens(string_to_lower(load_text(name)))))
    unique_n = unique_words(text)
    print('unique_words----->' + str(unique_n))
    mapping = create_integers(text)
    reverse_mapping = {i: k for k, i in mapping.items()}
    integers_texts = words_to_integers(text, mapping)
    ytm_batch = divider(integers_texts, 20, 30, 30)
    net = LSTM_Language_Model(27597, 100, 100, 2, 0.25)
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    train_LSTM(integers_texts, net, optimizer, 100, train=True)
    return net



def pre_process_valid_test_data_LSTM_upgrade(model, name='wiki.valid.txt', is_LSTM=True):
    setup_nltk()
    sliding_window_value = 30
    text = to_number(lists_to_tokens(splitting_tokens(string_to_lower(load_text(name)))))
    unique_n = unique_words(text)
    print('unique_words----->' + str(unique_n))
    mapping = create_integers(text)
    reverse_mapping = {i: k for k, i in mapping.items()}
    integers_texts = words_to_integers(text, mapping)
    valid(validation_data, model)



def train_LSTM(data, model, optimizer, clip_grads, epoch_size=3, train=False):
    if train == True:
        for i in range(3):
            model.train()
            for index, sequence in enumerate(divider(data, 20)):
                x = nn.utils.rnn.pack_sequence([torch.tensor(token[:-1]) for token in sequence])
                y = nn.utils.rnn.pack_sequence([torch.tensor(token[1:]) for token in sequence])
                model.zero_grad()
                out = model(x)
                loss = F.nll_loss(out, y.data)
                loss.backward()
                if clip_grads:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                if index % 150 == 0:

                    perplexity = np.exp(loss.item())
                    print("Batch" + ' ' + str(index))
                    print("loss" + ' ' + str(loss.item()))
                    if loss.item() < 6.9:
                        print('perplexity' + ' ' + str(perplexity))
                        break





def valid(data, model):
            model.eval()
            for index, sequence in enumerate(divider(data, 20)):
                x = nn.utils.rnn.pack_sequence([torch.tensor(token[:-1]) for token in sequence])
                y = nn.utils.rnn.pack_sequence([torch.tensor(token[1:]) for token in sequence])
                model.zero_grad()
                out = model(x)
                loss = F.nll_loss(out, y.data)
                loss.backward()
                if index % 150 == 0:
                    perplexity = np.exp(loss.item())
                    print("Batch" + ' ' + str(index))
                    print("loss" + ' ' + str(loss.item()))
                    print('perplexity' + ' ' + str(perplexity))

            print('final_perplexity_valid' + ' ' + str(perplexity))
            print("Batch" + ' ' + str(index))
            print("final_loss_valid" + ' ' + str(loss.item()))


