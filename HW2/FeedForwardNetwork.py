import torch.nn
from torch import nn
import time

class FeedForward(nn.Module):
    def __init__(self, input_size, number_of_classes, embedding_space):
        super(FeedForward, self).__init__()
        self.Linear1 = nn.Linear(input_size, embedding_space)#for bug of words we use number of classes as an put
        self.activation = torch.nn.Sigmoid()
        self.Linear2 = nn.Linear(embedding_space, number_of_classes)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.activation(self.Linear1(x))
        x = self.softmax(self.Linear2(x))

        return x
'''
    def init_weights(self):
        initrange = 0.1
        self.Linear1 =self.Linear1(-initrange, initrange)
'''


def train(dataloader, model, optimizer, criterion, epoch = 20):
    running_loss = 0
    model.train()


    for i, data in enumerate(dataloader):
        print(i)
        X,y = data
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions,y)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    print('Finished Training')