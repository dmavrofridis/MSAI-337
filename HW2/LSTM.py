import torch
from matplotlib import pyplot as plt
from torch import nn
import numpy as np
from preprocessing import *
from global_variables import *

class Module(nn.Module):
    def __init__(self, vocab_size=NUNBER_OF_CLASSES, n_class=NUNBER_OF_CLASSES, emb_dim=EMBEDDING_SPACE, hid=EMBEDDING_SPACE, num_layers=2, dropout=0.2):
        super(Module, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=emb_dim)
        self.linear1 = nn.Linear(hid, hid)
        self.linear2 = nn.Linear(hid, n_class)

        self.batchNorm = nn.BatchNorm1d(30)
        self.batchNorm2 = nn.BatchNorm1d(hid)
        self.dropout = nn.Dropout(0.1)

        self.hid = hid
        self.lstm = nn.LSTM(emb_dim, hid, num_layers, dropout=dropout, batch_first=True)
        self.n_class = n_class

        # self.linear2.weight = self.embedding.weight.T

    def init_weights(self):
        init_range = 0.1
        self.embedding = nn.Embedding(-init_range, init_range)
        self.linear1 = self.Linear(-init_range, init_range)
        self.linear2 = self.Linear(-init_range, init_range)

    def forward(self, seqs):
        batch_size, max_len = seqs.shape
        embs = self.embedding(seqs)

        embs = self.batchNorm(embs)
        output, (hidden, c) = self.lstm(embs)
        output = self.linear1(self.batchNorm2(output[:, -1, :]))
        output = self.linear2(output)
        return torch.log_softmax(output, dim=1)


def train(model, dataloader, optimizer, criterion, validation_dataloader, epoch=1, use_custom_loss=False):
    accuracy = []
    model.train()

    losses = []
    batches = []
    losses_to_visualize = []
    losses_to_visualize_valid = []


    for i in range(epoch):
        for i, data in enumerate(dataloader):

            X, y = data
            optimizer.zero_grad()
            predictions = model(X)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            loss = custom_cross_entropy_loss(predictions, y) if use_custom_loss else criterion(predictions, y)

            loss.backward()

            optimizer.step()

            if i % 100 == 0:
                losses.append(loss.item())
                losses_to_visualize.append(loss.item())
                print(i)
                print('mean_loss---------->' + ' ' + str(np.mean(losses)))
                if np.mean(losses) < 4.8:
                    break
                losses = []
            if i %2000 ==0 and i !=0:
                for j, data in enumerate(validation_dataloader):

                    total_valid = 0
                    correct_valid = 0
                    if j < 10000:
                        X, y = data
                        predictions = model(X)
                        loss_val = criterion(predictions, y)

                        losses_to_visualize_valid.append(loss_val.item())
                        _, predicted = torch.max(predictions.data, 1)
                        total_valid += y.size(0)
                        correct_valid += (predicted == y).sum().item()
                        if j == 10000:
                            print('validation_accuracy-FINAL---------------->' + str(100 * correct_valid// total_valid))
                            accuracy.append(100 * correct_valid // total_valid)

    plt.figure(figsize=(15, 15))
    print('train_loss------>' + str(losses_to_visualize))
    print(plt.plot(losses_to_visualize))
    print(plt.plot(accuracy))
    print('accuracy------>' + str(accuracy))
    print('perplexity--------------->' + ' ' + str(np.exp((loss.item()))))
    print('perplexity_second--------------->' + ' ' + str(np.exp((loss_val.item()))))
    print(plt.plot(losses_to_visualize_valid))
    print('validation_loss------>' + str(losses_to_visualize_valid))




