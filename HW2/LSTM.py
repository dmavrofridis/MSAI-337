import torch
from matplotlib import pyplot as plt
from torch import nn
import numpy as np
from preprocessing import *


class Module(nn.Module):
    def __init__(self, vocab_size=27597, n_class=27597, emb_dim=100, hid=100, num_layers=2, dropout=0.2):
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
    running_loss = 0
    accuracy = 0
    model.train()
    samples = 0
    trainAcc = 0
    losses = []
    batches = []
    losses_to_visualize = []

    for i in range(epoch):
        for i, data in enumerate(dataloader):

            X, y = data
            optimizer.zero_grad()
            predictions = model(X)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            loss = custom_cross_entropy_loss(predictions, y) if use_custom_loss else criterion(predictions, y)
            losses.append(loss.item())
            losses_to_visualize.append(loss.item())
            loss.backward()

            optimizer.step()

            if i % 100 == 0:
                print(i)
                print('mean_loss---------->' + ' ' + str(np.mean(losses)))
                if np.mean(losses) < 4.8:
                    break
                losses = []

                running_loss = 0.0
            trainAcc += (predictions.max(1)[1] == y).sum().item()
            samples += y.size(0)

    plt.figure(figsize=(15, 15))
    plt.plot(batches, losses_to_visualize)
    plt.plot(batches, accuracy)
    print('perplexity--------------->' + ' ' + str(losses_to_visualize(loss)))

    batches = []
    losses_to_visualize = []
    total = 0
    correct = 0

    total = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(validation_dataloader):
            if i > 10000:
                break
            X, y = data
            predictions = model(X)
            losses_to_visualize.append(loss.item())
            _, predicted = torch.max(predictions.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            if i % 1000 == 999:
                print('validation_accuracy------------->' + str(100 * correct // total))

        print('validation_accuracy-FINAL---------------->' + str(100 * correct // total))

    plt.plot(batches, accuracy)
    plt.show
    plt.plot(batches, losses_to_visualize)
    plt.show()
    print('perplexity--------------->' + ' ' + str(losses_to_visualize(loss)))
