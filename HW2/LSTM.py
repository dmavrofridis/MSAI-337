import torch
from torch import nn

import numpy as np
class Module(nn.Module):
    def __init__(self, vocab_size= 27597, n_class = 27597, emb_dim=100, hid=50, num_layers=1, dropout=0.2):
        super(Module, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=emb_dim)
        self.linear = nn.Linear(hid, hid)
        self.linear2 = nn.Linear(hid, hid)
        self.linear3 = nn.Linear(hid, n_class)


        self.batchNorm = nn.BatchNorm1d(30)
        self.batchNorm2 = nn.BatchNorm1d(hid)
        self.dropout = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU()
        self.hid = hid
        self.lstm = nn.LSTM(emb_dim, hid, num_layers, dropout=dropout, batch_first=True)
        self.n_class = n_class
    def init_weights(self):
        initrange = 0.1
        self.linear1 =self.Linear(-initrange, initrange)
        self.linear2 =self.Linear(-initrange, initrange)
        self.linear3 =self.Linear(-initrange, initrange)

    def forward(self, seqs):
        batch_size, max_len = seqs.shape
        embs = self.embedding(seqs)

        embs = self.batchNorm(embs)
        output, (hidden, c) = self.lstm(embs)
        output = self.linear(self.batchNorm2(output[:, -1, :]))
        output = self.leaky_relu( self.dropout(self.linear2(output)))
        output = self.linear3(output)
        return torch.log_softmax(output, dim=1)
    
def train(model, dataloader, optimizer, criterion,  validation_dataloader, epoch = 1,):

    running_loss = 0
    accuracy = 0
    model.train()
    samples = 0
    trainAcc = 0
    losses = []

    for i in range(epoch):
        for i, data in enumerate(dataloader):

            X,y = data
            optimizer.zero_grad()
            predictions = model(X)
            loss = criterion(predictions,y)
            losses.append(loss.item())

            loss.backward()

            optimizer.step()

            if i % 100 == 0:
                print(i)
                print('mean_loss---------->'+' '+ str(np.mean(losses)))
                if np.mean(losses) <4.9:
                    break
                losses = []

                running_loss = 0.0
            trainAcc += (predictions.max(1)[1] == y).sum().item()
            samples += y.size(0)


    total=0
    correct=0
    with torch.no_grad():
        for i, data in enumerate(validation_dataloader):
            if i > 10000:
                break
            X, y = data
            predictions = model(X)
            _, predicted = torch.max(predictions.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            if i %1000==999:
                print('validation_accuracy------------->' +  str(100 * correct // total))

        print('validation_accuracy-FINAL---------------->'+ str(100 * correct // total))
