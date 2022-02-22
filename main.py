import argparse
import time
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
    parser.add_argument('--num_worker', type=int, default=25,
                        help='number of dataloader worker')
    parser.add_argument('--h_dim', type=int, default=200,
                        help='size of hidden representation including embeddings')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Adam, AdamW, RMSprop, Adagrad, SGD')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--lr_step', type=int, default=10,
                        help='number of epoch for each lr downgrade')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='strength of lr downgrade')
    parser.add_argument('--epochs', type=int, default=50,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=1000, metavar='N',
                        help='batch size')
    parser.add_argument('--n_gram', type=int, default=40,
                        help='length of each training sequence')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--share_embedding', type=bool, default=True,
                        help='shared embedding and decoder weights')
    parser.add_argument('--share_embedding_strict', type=bool, default=True,
                        help='strictly shared embedding and decoder weights, no bias')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--es_patience_max', type=int, default=10,
                        help='Max early stopped patience')
    parser.add_argument('--save', type=str, default='./saved_model/model.pt',
                        help='path to save the final model')
    parser.add_argument('--onnx-export', type=str, default='./saved_model/model.onnx',
                        help='path to export the final model in onnx format')

    args = parser.parse_args()
    args.device = torch.device('cuda:0')
    print('\n[info] Project starts...')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # get data and prepare model, optimizer and scheduler
    my_dict, train_loader, valid_loader, test_loader = get_dataloader(args)
    args.n_token = len(my_dict)

    model = FNNModel(args).to(args.device)
    args.criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr,
                                     weight_decay=1e-4)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr,
                                      weight_decay=1e-4)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr,
                                        weight_decay=1e-4, momentum=0.9)
    elif args.optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr,
                                        weight_decay=1e-4)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr,
                                    weight_decay=1e-4, momentum=0.9, nesterov=True)
    else:
        raise RuntimeError('Wrong optimizer: '+args.optimizer+'...')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # Start modeling
    print('[info] | {optimizer} | N_gram {ngram:d} | Shared: {Shared} | Dropout {dropout:3.2f}| H_dim {h_dim:d} |'
          .format(optimizer=args.optimizer, ngram=args.n_gram, Shared=str(args.share_embedding), dropout=args.dropout,
                  h_dim=args.h_dim))
    best_val_loss = 1e5
    best_acc = 0
    es_patience = 0

    for epoch in range(1, args.epochs+1):
        print('\n[Epoch {epoch}]'.format(epoch=epoch))

        tr_loss, tr_acc = train(args, model, train_loader, optimizer)
        scheduler.step()
        val_loss, val_acc = evaluate(args, model, valid_loader, mode='Valid')

        # Save the model if the validation loss is the best we've seen so far.
        if val_loss <= best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            best_acc = val_acc
            es_patience = 0
        else:
            # Early stopping condition
            es_patience += 1
            if es_patience >= args.es_patience_max:
                print('\n[Warning] Early stopping model')
                print('  | Best  | loss {:5.4f} | ppl {:8.2f} | acc {:5.4f} |'
                      .format(best_val_loss, math.exp(best_val_loss), best_acc))
                break

    # Load the best saved model and test
    print('\n[Testing]')
    with open(args.save, 'rb') as f:
        model = torch.load(f)
    test_loss, test_acc = evaluate(args, model, test_loader, mode='Test')
    print('  | {optimizer} | N_gram {ngram:d} | Shared: {Shared} | Dropout {dropout:3.2f}| H_dim {h_dim:d} |'
          .format(optimizer=args.optimizer, ngram=args.n_gram, Shared=str(args.share_embedding), dropout=args.dropout,
                  h_dim=args.h_dim))

    # Export the model in ONNX format.
    if len(args.onnx_export) > 0:
        print('\n[info] The model is also exported in ONNX format at {}\n'.
              format(os.path.realpath(args.onnx_export)))
        model.eval()
        dummy_input = torch.LongTensor(args.n_gram * args.batch_size).zero_().view(-1, args.batch_size).to(args.device)
        torch.onnx.export(model, dummy_input, args.onnx_export)


if __name__ == '__main__':
    main()
