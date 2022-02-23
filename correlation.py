import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.onnx

from dataloader import get_dataloader
from scipy.stats import spearmanr


def main():
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
    parser.add_argument('--path_data', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device for modeling')

    parser.add_argument('--path_data_new', type=str, default='./data/wordsim353_sim_rel/'
                                                             'wordsim_similarity_goldstandard.txt',
                        help='location of the data corpus')
    parser.add_argument('--checkpoint', type=str, default='./saved_model/model.pt',
                        help='model checkpoint to use')

    # prepare hyperparameters and model
    args = parser.parse_args()
    args.device = torch.device(args.device)
    print('\n[info] Computing correlation starts...')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f).to(args.device)
    model.eval()

    # get data new
    my_dict = get_dataloader(args, no_dataloader=True)
    keys_my_dict = my_dict.word2idx.keys()
    args.n_token = len(my_dict)

    word_1, word_2, sim_new = [], [], []
    with open(args.path_data_new, 'r') as f:
        for line in f.readlines():
            line = line.split()
            if line[0] in keys_my_dict and line[1] in keys_my_dict:
                word_1.append(my_dict.word2idx[line[0]])
                word_2.append(my_dict.word2idx[line[1]])
                sim_new.append(float(line[2]))

    word_1 = torch.LongTensor(word_1).to(args.device)
    word_2 = torch.LongTensor(word_2).to(args.device)
    sim_new = np.array(sim_new)

    # get embedding cosine similarity
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    sim_old = cos(model.encoder(word_1), model.encoder(word_2)).detach().cpu().numpy()
    cor = spearmanr(sim_new, sim_old)[0]
    print('  | Spearman correlation = {cor:.4f}'.format(cor=cor))


if __name__ == '__main__':
    main()
