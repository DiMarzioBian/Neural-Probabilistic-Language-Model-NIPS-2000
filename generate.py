import argparse
import math
import os
import torch
import torch.nn as nn
import torch.onnx

from dataloader import get_dataloader
from model import FNNModel
from epoch import train, evaluate


def main():
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
    parser.add_argument('--path_data', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--n_gram', type=int, default=40,
                        help='length of each training sequence')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device for modeling')

    parser.add_argument('--out_f', type=str, default='./result/generated.txt',
                        help='output file for generated text')
    parser.add_argument('--n_words', type=int, default=100,
                        help='number of generated words')
    parser.add_argument('--checkpoint', type=str, default='./saved_model/model.pt',
                        help='model checkpoint to use')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature - higher will increase diversity')
    parser.add_argument('--log-interval', type=int, default=20,
                        help='reporting interval')

    # prepare hyperparameters and model
    args = parser.parse_args()
    args.device = torch.device(args.device)
    print('\n[info] Generation starts...')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.temperature < 1e-3:
        parser.error("--temperature has to be greater or equal 1e-3")

    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f).to(args.device)
    model.eval()

    # get data
    my_dict = get_dataloader(args, no_dataloader=True)
    args.n_token = len(my_dict)

    # Set lyrics as input from <Numb - Linkin Park>
    input_words = "I 'm tired of being what you want me to be . <eos> Feeling so sad , lost under the surface . <eos>" \
                  " Dont know what you are expecting of me . <eos> Put under the pressure of walking in"
    print()
    input_idx = []

    with open(args.out_f, 'w') as out_f:
        # print input in txt file
        out_f.write('-' * 40 + 'Prediction' + '-' * 40 + '\n')
        for i, word in enumerate(input_words.split()[-args.n_gram:]):
            out_f.write(word + ('\n' if i % 20 == 19 else ' '))
            input_idx.append(my_dict.word2idx[word])
        input_idx = torch.LongTensor(input_idx).unsqueeze(0).to(args.device)
        out_f.write('\n' + '-' * 40 + 'Prediction' + '-' * 40 + '\n')

        # print prediction in txt file
        with torch.no_grad():
            for i in range(args.n_words):
                word_weights = model(input_idx).squeeze(0).div(args.temperature).exp()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input_idx = torch.cat((input_idx[0, 1:], word_idx.unsqueeze(0)), dim=0).unsqueeze(0)

                word = my_dict.idx2word[word_idx]

                out_f.write(word + ('\n' if i % 20 == 19 else ' '))

                if i % args.log_interval == 19:
                    print('| Generated {}/{} words'.format(i+1, args.n_words))


if __name__ == '__main__':
    main()
