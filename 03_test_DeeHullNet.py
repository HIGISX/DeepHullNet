import torch
import os
import sys
import importlib
from process import Scatter2DDataset
from torch.utils.data import DataLoader
from cal import AverageMeter, masked_accuracy, calculate_hull_overlap
import numpy as np
import argparse

TOKENS = {
    '<sos>': 0,
    '<eos>': 1
}

def build_data_loader(dataset_path='./dataset/'):
    dataset_path = dataset_path

    test_dataset = Scatter2DDataset(file_name=dataset_path+'test_dateset.json')

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=True,
        num_workers=0, pin_memory=False)

    print('Got {} test samples'.format(len(test_loader.dataset)))
    return test_loader

def measure(model, test_loader, device):
    n_per_row = 50
    test_accuracy = AverageMeter()
    Seq_accuracy = AverageMeter()
    hull_overlaps = []
    model.eval()

    for bat, (batch_data, batch_labels, batch_lengths) in enumerate(test_loader):
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        batch_lengths = batch_lengths.to(device)

        log_pointer_scores, pointer_argmaxs = model(batch_data, batch_lengths, batch_labels=batch_labels)
        mask = batch_labels != TOKENS['<eos>']
        acc, seq_acc = masked_accuracy(pointer_argmaxs, batch_labels, mask)
        test_accuracy.update(acc.item(), mask.int().sum().item())
        Seq_accuracy.update(seq_acc.item())
        for data, length, ptr in zip(batch_data.cpu(), batch_lengths.cpu(),
                                     pointer_argmaxs.cpu()):
            hull_overlaps.append(calculate_hull_overlap(data, length, ptr))

    print(f'# Test Samples: {n_per_row:3d}\t nodes'
          f'\tAccuracy: {test_accuracy.avg:3.1%} '
          # f'\tSeq_accuracy: {Seq_accuracy.avg:3.1%} '
          f'\tOverlap: {np.mean(hull_overlaps):3.1%}')
    return test_accuracy, Seq_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN train and validate')
    parser.add_argument(
        'problem',
        help='instance type to process.',
        choices=['convex', 'concave'],
    )
    parser.add_argument(
        '-m', '--model',
        help='model to be trained.',
        type=str,
        default='Transform',
        choices=['Linear', 'LSTM']
    )

    args = parser.parse_args()
    c_inputs = 5
    c_embed = 16
    c_hidden = 16
    n_heads = 4
    n_layers = 3
    dropout = 0.0
    use_cuda = True

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    ### MODEL LOADING ###
    sys.path.insert(0, os.path.abspath(f'models/{args.model}'))
    import model
    importlib.reload(model)
    model = model.DeepHullNet(c_inputs=c_inputs, c_embed=c_embed, n_heads=n_heads,
                      n_layers=n_layers, dropout=dropout, c_hidden=c_hidden).to(device)
    del sys.path[0]
    if args.problem == 'convex':
        checkpoint_file = f'./checkpoint/{args.problem}/50-50/{args.model}/best_params.pkl'
        model.load_state_dict(torch.load(checkpoint_file)['model_state'])
        dataset_path = f'./dataset/{args.problem}/50-50/'
        test_loader = build_data_loader(dataset_path)
        measure(model, test_loader, device)

    elif args.problem == 'concave':
        checkpoint_file = f'./checkpoint/{args.problem}/50-50/{args.model}/best_params.pkl'
        model.load_state_dict(torch.load(checkpoint_file)['model_state'])
        dataset_path = f'./dataset/{args.problem}/50-50/'
        test_loader = build_data_loader(dataset_path)
        measure(model, test_loader, device)

