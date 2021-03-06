import os
from io import open
import torch
from torch.utils.data import Dataset, DataLoader

# import numpy as np
# import matplotlib.pyplot as plt


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
            idss += ids

    return dictionary, idss


class WikiTextData(Dataset):
    """ WikiText Dataset """
    def __init__(self, args, tks_file):
        self.initial_preprocess = args.initial_preprocess
        self.n_gram = args.n_gram
        self.tokens_file = tks_file
        if self.initial_preprocess:
            self.length = len(tks_file) // (args.n_gram+1)
        else:
            self.length = len(tks_file) - args.n_gram

        # EDA: plot frequency
        # tks = np.array(tks_file)
        # unique, counts = np.unique(tks, return_counts=True)
        # counts = counts[::-1]
        #
        # fig, axs = plt.subplots(2, 1)
        # axs[0].plot(counts)
        # axs[0].set_ylabel('Frequency')
        # axs[0].grid(True)
        #
        # axs[1].plot(np.log(counts))
        # axs[1].set_ylabel('Frequency in log')
        # axs[1].set_xlabel('Words / tokens')
        # axs[1].grid(True)
        #
        # fig.tight_layout()
        # plt.show()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.initial_preprocess:
            return self.tokens_file[index * self.n_gram: (index+1) * self.n_gram], \
                   self.tokens_file[(index+1) * self.n_gram]
        else:
            return self.tokens_file[index: index+self.n_gram], self.tokens_file[index+self.n_gram]


def collate_fn(insts):
    """ Batch preprocess """
    seq_tokens_batch, tgt_tokens_batch = list(zip(*insts))

    seq_tokens_batch = torch.LongTensor(seq_tokens_batch)
    tgt_tokens_batch = torch.LongTensor(tgt_tokens_batch)
    return seq_tokens_batch, tgt_tokens_batch


def get_dataloader(args, no_dataloader=False):
    """ Get dataloader and dictionary """
    my_dict = Dictionary()

    my_dict, train_data = tokenize(my_dict, path=os.path.join(args.path_data, 'train.txt'))
    my_dict, valid_data = tokenize(my_dict, path=os.path.join(args.path_data, 'valid.txt'))
    my_dict, test_data = tokenize(my_dict, path=os.path.join(args.path_data, 'test.txt'))

    if no_dataloader:
        # For generation and similarity calculation quest which do not need dataloader
        return my_dict

    train_loader = DataLoader(WikiTextData(args, train_data), batch_size=args.batch_size,
                              num_workers=args.num_worker, collate_fn=collate_fn, shuffle=True)
    valid_loader = DataLoader(WikiTextData(args, valid_data), batch_size=args.batch_size,
                              num_workers=args.num_worker, collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(WikiTextData(args, test_data), batch_size=args.batch_size, num_workers=args.num_worker,
                             collate_fn=collate_fn, shuffle=True)
    return my_dict, train_loader, valid_loader, test_loader

