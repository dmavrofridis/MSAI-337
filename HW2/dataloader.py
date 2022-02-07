import torch
def overall_words(text):
    return len(text)


def unique_words(text):
    return len(set(text))




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

class wikiDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        #self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        print(idx)
        sample = torch.Tensor(self.data[idx])

        return sample

def  batch_divder(vectors, batch_size = 20):
    data =  torch.utils.data.DataLoader(vectors,batch_size)
    return data














