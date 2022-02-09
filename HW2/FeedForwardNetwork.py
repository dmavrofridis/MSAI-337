import torch.nn
from torch import nn
import time

class FeedForward(nn.Module):
    def __init__(self, window_size, number_of_classes, embedding_space, tie_weights=True):
        super(FeedForward, self).__init__()
        self.encode = nn.Embedding(number_of_classes, embedding_space)
        self.Linear1 = nn.Linear(embedding_space, embedding_space)#for bug of words we use number of classes as an put
        self.activation = torch.nn.ReLU()
        self.Linear2 = nn.Linear(embedding_space, number_of_classes)
        self.softmax = torch.nn.Softmax()

        if tie_weights:
            self.Linear2.weight = self.encode.weight

    def forward(self, x):
        #print("Current x:", x)
        embeds = self.encode(x) #.view((1,-1))
        out = self.activation(self.Linear1(embeds))
        out = self.Linear2(out)

        return out
'''
    def init_weights(self):
        initrange = 0.1
        self.Linear1 =self.Linear1(-initrange, initrange)
'''


def train(dataloader, model, optimizer, criterion, epoch = 20):
    running_loss = 0
    accuracy = 0
    model.train()
    samples = 0
    trainAcc = 0


    for i, data in enumerate(dataloader):
        X,y = data
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions,y)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 1000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}')
            print('Accuracy------> ' + str(trainAcc / samples))


            running_loss = 0.0
        trainAcc += (predictions.max(1)[1] == y).sum().item()
        samples += y.size(0)

    print('Finished Training')
    print('Accuracy------> '+ str(trainAcc / samples))