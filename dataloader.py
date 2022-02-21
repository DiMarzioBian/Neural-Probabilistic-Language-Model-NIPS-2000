import os
from io import open
import torch
from torch.utils.data import Dataset, DataLoader


# Dictionary to store all words and their indices
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


# Tokenize given texts
def tokenize(dictionary, path):
    """Tokenizes a text file."""
    assert os.path.exists(path)
    # Add words to the dictionary
    with open(path, 'r', encoding="utf8") as f:
        for line in f:
            words = line.split() + ['<eos>']
            for word in words:
                dictionary.add_word(word)

    # Tokenize file content
    with open(path, 'r', encoding="utf8") as f:
        idss = []
        for line in f:
            words = line.split() + ['<eos>']
            ids = []
            for word in words:
                ids.append(dictionary.word2idx[word])
            idss.append(torch.IntTensor(ids))
        ids = torch.cat(idss)

    return ids


class WikiTextData(Dataset):
    """WikiText Dataset"""
    def __init__(self, args, tks_file):
        self.item_seq, self.item_gt, self.date_seq, self.date_gt = data
        self.length = len(self.item_seq)

    def __len__(self):
        return self.length

    def __getitem__(self, index):



# Get dataloader for train and valid and test set
def get_dataloader(args):
    my_dict = Dictionary()

    train_data = tokenize(my_dict, path='data/train.txt')
    valid_data = tokenize(my_dict, path='data/valid.txt')
    test_data = tokenize(my_dict, path='data/test.txt')

    train_loader = DataLoader(WikiTextData(args, train_data), batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=True)
    valid_loader = DataLoader(WikiTextData(args, valid_data), batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=True)
    test_loader = DataLoader(WikiTextData(args, test_data), batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=True)

    return my_dict, train_loader, valid_loader, test_loader

