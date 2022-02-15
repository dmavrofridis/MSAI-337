import torch.nn
from torch import nn
import time
import numpy as np
from global_variables import *

def softmax(x):
    exp_x = torch.exp(x)
    sum_x = torch.sum(exp_x, dim=1, keepdim=True)
    return exp_x / sum_x
def log_softmax(x):
    return x - torch.logsumexp(x,dim=1, keepdim=True)
def custom_cross(outputs, targets):
    num_examples = targets.shape[0]
    batch_size = outputs.shape[0]
    outputs = log_softmax(outputs)
    outputs = outputs[range(batch_size), targets]
    return - torch.sum(outputs) / num_examples
class FeedForward(nn.Module):
    def __init__(self, input_size, number_of_classes, embedding_space, window_size=5):
        super(FeedForward, self).__init__()
        self.embed = nn.Embedding(27597, 100)
        self.Linear1 = nn.Linear(100 * window_size, embedding_space)
        self.batchNorm = nn.BatchNorm1d(100)
        self.activation = torch.nn.ReLU()
        self.Linear2 = nn.Linear(embedding_space, number_of_classes)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        embeds = self.embed(x)
        x = embeds.view(-1, embeds.size(1) * embeds.size(2))
        x = self.activation(self.batchNorm(self.Linear1(x)))
        x = self.Linear2(x)

        return x

    def init_weights(self):
        initrange = 0.1
        self.Linear1 = self.Linear1(-initrange, initrange)


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


def train(dataloader, model, optimizer, criterion, validation_dataloader, custom_loss = True, epoch=2):
    running_loss = 0
    accuracy = 0
    model.train()
    samples = 0
    trainAcc = 0
    losses = []

    for epoch in range(epoch):
        for index, data in enumerate(dataloader):

            X, y = data
            optimizer.zero_grad()
            predictions = model(X)
            if custom_loss == False:
                loss = criterion(predictions, y)
            else:
                loss = custom_cross(predictions, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if index % 1000 == 0:
                print(index)
                print('mean_loss---------->' + ' ' + str(np.mean(losses)))
                if np.mean(losses) < 6.8:
                    break
                losses = []
                running_loss = 0.0

            trainAcc += (predictions.max(1)[1] == y).sum().item()
            samples += y.size(0)

    total = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(validation_dataloader):
            if i > 10000:
                break
            X, y = data
            predictions = model(X)
            _, predicted = torch.max(predictions.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            if i % 1000 == 999:
                print('validation_accuracy------------->' + str(100 * correct // total))

        print('validation_accuracy-FINAL---------------->' + str(100 * correct // total))

    print('Finished Training')
