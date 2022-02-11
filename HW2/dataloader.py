import torch
import numpy as np

def overall_words(text):
    return len(text)


def unique_words(text):
    return len(set(text))


def create_integers(text):
    words_integers = {}
    current = 0
    for word in text:
        if word not in words_integers:
            words_integers[word] = current
            current +=1
    return words_integers


def words_to_integers(text,mapping):
    final =[]
    for word in text:
        if word not in mapping:
            word = '<unk>'
        final.append(mapping[word])

    return final




def create_one_hot_encoddings(text, unique_words_length):
    one_hot_dics = {}
    index = 0
    new_vec= [0 for i in range( unique_words_length) ]

    for word in text:
        if word not in one_hot_dics:
            new_vec[index] =1
            one_hot_dics[word] = list(new_vec)
            new_vec[index]=0
            index +=1
    return one_hot_dics


def map_words_to_vec(text, vectors):
    one_hot_vectors=[]
    for word in text:
        one_hot_vectors.append(vectors[word])
    return one_hot_vectors

#transform = torchvision.transforms.Compose(torchvision.transforms.ToTensor())

def sliding_window(integer_list, window_size):
    slised =[]
    current =[]
    for i in range(len(integer_list)):
        current.append(integer_list[i])
        if (i+1)>= window_size:
            slised.append(current)
            current = current[1:]


    return slised

def label_generation(integer_list):
    labels =[]
    for i in range(len(integer_list)):
        if i >=5:
            labels.append(integer_list[i])
    return labels





class wikiDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        #self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = torch.Tensor(self.data[idx]).long()
        label = torch.Tensor(self.labels[idx]).long()

        label = (torch.argmax(label).long())

        return sample, label



class wikiDatasetBagOfWords(torch.utils.data.Dataset):
    def __init__(self, data, labels, reverse_mapping, one_hot_vectors):
        self.data = data
        self.labels = labels
        self.reverse_mapping = reverse_mapping
        self.one_hot_vectors =one_hot_vectors
        #self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        bag = integers_to_vectors(sample, self.reverse_mapping, self.one_hot_vectors)
        X = torch.Tensor(sum([np.array(i) for i in bag]))

        label = torch.Tensor(self.labels[idx])

        label = torch.argmax(label)
        return X, label


def  batch_divder(vectors, batch_size = 20):
    data =  torch.utils.data.DataLoader(vectors,batch_size)
    return data


def integers_to_vectors(integers, reverse_mapping, one_hot_vectors):
    res = []
    for num in integers:
        word = reverse_mapping[num]
        vec = one_hot_vectors[word]
        res.append(vec)
    return res

















