from torch import nn
import torch
from global_variables import EMBEDDING_SIZE, WINDOW

class FeedForwardText(nn.Module):

    def __init__(self, vocab_size, embedding_size, tie_weights=True):
        super(FeedForwardText,self).__init__()
        self.encode = nn.Embedding(vocab_size, embedding_size)
        self.ll1 = nn.Linear(embedding_size, embedding_size)
        self.act = nn.ReLU()
        self.decode = nn.Linear(embedding_size, vocab_size)

        if tie_weights:
            self.decode.weight = self.encode.weight

    def forward(self,x):
        embed = self.encode(x)  #.view((1,-1))
        out = self.act(self.ll1(embed))
        out = self.decode(out)
        return out

    #add init_weights

def train(dataloader, model, optimizer, loss_fn, num_epochs = 20):
    running_loss = 0
    accuracy = 0
    model.train()
    samples = 0
    for i in range(num_epochs):
        for index, batch in enumerate(dataloader):

            labels = []
            data = []
            for group in batch:

                X,y = group
                labels.append(y)
                data.append(X)


            input = torch.tensor(data)
            labels = torch.tensor(labels)

            optimizer.zero_grad()
            predictions = model(input)

            loss = loss_fn(predictions, labels)

            running_loss += loss.item()
            if index % 1000 == 999:  # print every 1000 mini-batches
                print(f'[{i + 1}, {index + 1:5d}] loss: {running_loss / 1000:.3f}')
                print('Accuracy------> ' + str(accuracy / samples))  #by the way, I despise this print formatting

                running_loss = 0.0
            accuracy += (predictions.max(1)[1] == labels).sum().item()
            samples += labels.size(0) * WINDOW * EMBEDDING_SIZE



