import torch.nn
from torch import nn
import time
import numpy as np
from preprocessing import *
from global_variables import *
import matplotlib.pyplot as plt

class FeedForward(nn.Module):
    def __init__(self, input_size=27597, number_of_classes=27597, embedding_space=100, window_size=5):
        super(FeedForward, self).__init__()
        self.embed = nn.Embedding(number_of_classes, embedding_space)
        self.Linear1 = nn.Linear(embedding_space * window_size, embedding_space)
        self.batchNorm = nn.BatchNorm1d(embedding_space)
        self.activation = torch.nn.Tanh()
        self.Linear2 = nn.Linear(embedding_space, number_of_classes)
        self.softmax = torch.nn.Softmax()
        #tight_embeddings =
        #self.Linear2.weights = self.embed.weights.T

    def forward(self, x):
        embeds = self.embed(x)
        x = embeds.view(-1, embeds.size(1) * embeds.size(2))
        x = self.activation(self.batchNorm(self.Linear1(x)))
        x = self.Linear2(x)

        return x

    def init_weights(self):
        initrange = 0.1
        self.embed = self.embed(-initrange, initrange)
        self.Linear1 = self.Linear1(-initrange, initrange)
        self.Linear2 = self.Linear2(-initrange, initrange)


class FeedForwardText(nn.Module):

    def __init__(self, vocab_size, embedding_size, tie_weights=True):
        super(FeedForwardText, self).__init__()
        self.encode = nn.Embedding(27597, 100)
        self.ll1 = nn.Linear(100, 100)
        self.act = nn.ReLU()
        self.decode = nn.Linear(embedding_size, vocab_size)

        if tie_weights:
            self.decode.weight = self.encode.weight

    def forward(self, x):
        embed = self.encode(x)  # .view((1,-1))
        out = self.act(self.ll1(embed))
        out = self.decode(out)
        return out


def train(model, dataloader, optimizer, criterion, validation_dataloader, epoch=1, use_custom_loss=False):
    accuracy = []
    model.train()

    losses = []
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
                losses_to_visualize.append(loss.item())
                print(i)
                print('mean_loss---------->' + ' ' + str(loss.item()))
                if loss.item() < 5.3:
                    for j, data in enumerate(validation_dataloader):

                        total_valid = 0
                        correct_valid = 0
                        if j < 10000:
                            X, y = data
                            predictions = model(X)
                            loss_val = criterion(predictions, y)

                            _, predicted = torch.max(predictions.data, 1)
                            total_valid += y.size(0)
                            correct_valid += (predicted == y).sum().item()
                            if j == 9000:
                                print('validation_accuracy-FINAL---------------->' + str(
                                    100 * correct_valid // total_valid))
                                accuracy.append(100 * correct_valid // total_valid)
                                losses_to_visualize_valid.append(loss_val.item())
                    break
                losses = []
            if i %500 ==0 and i !=0:
                for j, data in enumerate(validation_dataloader):

                    total_valid = 0
                    correct_valid = 0
                    if j < 10000:
                        X, y = data
                        predictions = model(X)
                        loss_val = criterion(predictions, y)

                        _, predicted = torch.max(predictions.data, 1)
                        total_valid += y.size(0)
                        correct_valid += (predicted == y).sum().item()
                        if j == 9000:
                            print('validation_accuracy-FINAL---------------->' + str(100 * correct_valid// total_valid))
                            accuracy.append(100 * correct_valid // total_valid)
                            losses_to_visualize_valid.append(loss_val.item())

    plt.figure(figsize=(15, 15))
    print('train_loss------>' + str(losses_to_visualize))
    print(plt.plot(losses_to_visualize))
    print(plt.plot(accuracy))
    print('accuracy------>' + str(accuracy))
    print('perplexity--------------->' + ' ' + str(np.exp((loss.item()))))
    print('perplexity_second--------------->' + ' ' + str(np.exp((loss_val.item()))))
    print(plt.plot(losses_to_visualize_valid))
    print('validation_loss------>' + str(losses_to_visualize_valid))
