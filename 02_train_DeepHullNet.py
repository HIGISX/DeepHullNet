import numpy as np
from process import Scatter2DDataset
from torch.utils.data import DataLoader
import torch
import argparse
import os
from cal import AverageMeter, masked_accuracy, calculate_hull_overlap
from tensorboardX import SummaryWriter
import sys
import importlib
TOKENS = {
    '<sos>': 0,
    '<eos>': 1
}
def build_data_loader(dataset_path='./dataset/'):
    dataset_path = dataset_path
    train_dataset = Scatter2DDataset(file_name=dataset_path+'train_dateset.json')
    val_dataset = Scatter2DDataset(file_name=dataset_path+'val_dateset.json')

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=False)

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=False)

    print('Got {} training samples'.format(len(train_loader.dataset)))
    print('Got {} validation samples'.format(len(val_loader.dataset)))
    return train_loader, val_loader

def train(model, train_loader, device, optimizer, epoch, tb_writer, total_tb_it):
    model.train()

    for bat, (batch_data, batch_labels, batch_lengths) in enumerate(train_loader):
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        batch_lengths = batch_lengths.to(device)

        optimizer.zero_grad()
        log_pointer_scores, pointer_argmaxs = model(batch_data, batch_lengths, batch_labels=batch_labels)
        loss = criterion(log_pointer_scores.view(-1, log_pointer_scores.shape[-1]), batch_labels.reshape(-1))
        assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), batch_data.size(0))
        mask = batch_labels != TOKENS['<eos>']
        acc, seq_acc = masked_accuracy(pointer_argmaxs, batch_labels, mask)
        train_accuracy.update(acc.item(), mask.int().sum().item())
        Seq_accuracy.update(seq_acc.item())
        per_loss = loss.item() / batch_data.size(0)

        tb_writer.add_scalar('train/overall_loss', per_loss, total_tb_it)
        tb_writer.add_scalar('train/accuracy', acc, total_tb_it)
        total_tb_it += 1

        if bat % log_interval == 0:
            print(f'Epoch {epoch}: '
                  f'Train [{bat * len(batch_data):9d}/{len(train_loader.dataset):9d} '
                  # f'\tSeq_accuracy: {Seq_accuracy.avg:3.1%} '
                  f'Loss: {train_loss.avg:.6f}\tAccuracy: {train_accuracy.avg:3.4%} ')
    return train_loss, train_accuracy

def validata(model, val_loader, device, tb_writer, total_tb_it):
    model.eval()

    hull_overlaps = []
    for bat, (batch_data, batch_labels, batch_lengths) in enumerate(val_loader):
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        batch_lengths = batch_lengths.to(device)

        log_pointer_scores, pointer_argmaxs = model(batch_data, batch_lengths,
                                                    batch_labels=batch_labels)
        loss = criterion(log_pointer_scores.view(-1, log_pointer_scores.shape[-1]), batch_labels.reshape(-1))
        assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

        val_loss.update(loss.item(), batch_data.size(0))
        mask = batch_labels != TOKENS['<eos>']
        acc, seq_acc = masked_accuracy(pointer_argmaxs, batch_labels, mask)
        val_accuracy.update(acc.item(), mask.int().sum().item())
        Seq_accuracy.update(seq_acc.item())
        per_loss = loss.item() / batch_data.size(0)

        tb_writer.add_scalar('val/overall_loss', per_loss, total_tb_it)
        tb_writer.add_scalar('val/accuracy', acc, total_tb_it)
        total_tb_it += 1

        for data, length, ptr in zip(batch_data.cpu(), batch_lengths.cpu(),
                                     pointer_argmaxs.cpu()):
            hull_overlaps.append(calculate_hull_overlap(data, length, ptr))

    print(f'Epoch {epoch}: Val\tLoss: {val_loss.avg:.6f} '
          f'\tAccuracy: {val_accuracy.avg:3.4%} '
          # f'\tSeq_accuracy: {Seq_accuracy.avg:3.1%} '
          f'\tOverlap: {np.mean(hull_overlaps):3.4%}')
    return val_loss

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
    ##define parameters
    workers = 4
    c_inputs = 5
    c_embed = 16
    c_hidden = 16
    n_heads = 4
    n_layers = 3
    dropout = 0.0
    use_cuda = True

    n_epochs = 2000
    batch_size = 16
    lr = 1e-3
    patience = 10
    early_stopping = 20
    log_interval = 125

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    ### MODEL LOADING ###
    sys.path.insert(0, os.path.abspath(f'models/{args.model}'))
    import model
    importlib.reload(model)
    model = model.DeepHullNet(c_inputs=c_inputs, c_embed=c_embed, n_heads=n_heads,
                      n_layers=n_layers, dropout=dropout, c_hidden=c_hidden).to(device)
    del sys.path[0]

    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    val_loss = AverageMeter()
    val_accuracy = AverageMeter()
    Seq_accuracy = AverageMeter()
    best_loss = np.inf
    total_tb_it = 0

    if args.problem == 'convex':
        checkpoint_dir = f'./checkpoint/convex/50-50/{args.model}/'
        model_name = checkpoint_dir + 'best_params.pkl'
        os.makedirs(os.path.dirname(model_name), exist_ok=True)
        log_folder = f'./tb-logs/convex/50-50/{args.model}/'
        writer = SummaryWriter(log_folder)

        ### load the data ###
        dataset_path = './dataset/convex/50-50/'
        train_loader, val_loader = build_data_loader(dataset_path=dataset_path)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.NLLLoss(ignore_index=-1)
        for epoch in range(n_epochs):
            train_loss, train_accuracy = train(model, train_loader, device, optimizer, epoch, writer, total_tb_it)
            val_loss = validata(model, val_loader, device, writer, total_tb_it)

            if val_loss.avg < best_loss:
                plateau_count = 0
                best_loss = val_loss.avg
                state = {'epoch': epoch, 'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict()}
                torch.save(state, model_name)
                print('Best model so far')
            else:
                plateau_count += 1
                if plateau_count % early_stopping == 0:
                    print(f"  {plateau_count} epochs without improvement, early stopping")
                    break
                if plateau_count % patience == 0:
                    lr *= 0.2
                    print(f"  {plateau_count} epochs without improvement, decreasing learning rate to {lr}")
            train_loss.reset()
            train_accuracy.reset()
            val_loss.reset()
            val_accuracy.reset()
        writer.close()

        hull_overlaps = []
        for bat, (batch_data, batch_labels, batch_lengths) in enumerate(val_loader):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            batch_lengths = batch_lengths.to(device)

            log_pointer_scores, pointer_argmaxs = model(batch_data, batch_lengths,
                                                        batch_labels=batch_labels)
            loss = criterion(log_pointer_scores.view(-1, log_pointer_scores.shape[-1]), batch_labels.reshape(-1))
            assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

            val_loss.update(loss.item(), batch_data.size(0))
            mask = batch_labels != TOKENS['<eos>']
            acc, seq_acc = masked_accuracy(pointer_argmaxs, batch_labels, mask)
            val_accuracy.update(acc.item(), mask.int().sum().item())

            for data, length, ptr in zip(batch_data.cpu(), batch_lengths.cpu(),
                                         pointer_argmaxs.cpu()):
                hull_overlaps.append(calculate_hull_overlap(data, length, ptr))
        print(f'BEST Val\tLoss: {val_loss.avg:.6f} '
              f'\tAccuracy: {val_accuracy.avg:3.4%} '
              # f'\tSeq_accuracy: {Seq_accuracy.avg:3.1%} '
              f'\tOverlap: {np.mean(hull_overlaps):3.4%}')

    elif args.problem == 'concave':

        train_loss = AverageMeter()
        train_accuracy = AverageMeter()
        val_loss = AverageMeter()
        val_accuracy = AverageMeter()
        Seq_accuracy = AverageMeter()
        best_loss = np.inf

        checkpoint_dir = f'./checkpoint/concave/50-50/{args.model}/'
        model_name = checkpoint_dir + 'best_params.pkl'
        os.makedirs(os.path.dirname(model_name), exist_ok=True)
        log_folder = f'./tb-logs/concave/50-50/{args.model}/'
        writer = SummaryWriter(log_folder)

        ### load the data ###
        dataset_path = './dataset/concave/50-50/'
        train_loader, val_loader = build_data_loader(dataset_path=dataset_path)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.NLLLoss(ignore_index=-1)
        for epoch in range(n_epochs):
            train_loss, train_accuracy = train(model, train_loader, device, optimizer, epoch, writer, total_tb_it)
            val_loss = validata(model, val_loader, device, writer, total_tb_it)

            if val_loss.avg < best_loss:
                plateau_count = 0
                best_loss = val_loss.avg
                state = {'epoch': epoch, 'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict()}
                torch.save(state, model_name)
                print('Best model so far')
            else:
                plateau_count += 1
                if plateau_count % early_stopping == 0:
                    print(f"  {plateau_count} epochs without improvement, early stopping")
                    break
                if plateau_count % patience == 0:
                    lr *= 0.2
                    print(f"  {plateau_count} epochs without improvement, decreasing learning rate to {lr}")
            train_loss.reset()
            train_accuracy.reset()
            val_loss.reset()
            val_accuracy.reset()
        writer.close()

        hull_overlaps = []
        for bat, (batch_data, batch_labels, batch_lengths) in enumerate(val_loader):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            batch_lengths = batch_lengths.to(device)

            log_pointer_scores, pointer_argmaxs = model(batch_data, batch_lengths,
                                                        batch_labels=batch_labels)
            loss = criterion(log_pointer_scores.view(-1, log_pointer_scores.shape[-1]), batch_labels.reshape(-1))
            assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

            val_loss.update(loss.item(), batch_data.size(0))
            mask = batch_labels != TOKENS['<eos>']
            acc, seq_acc = masked_accuracy(pointer_argmaxs, batch_labels, mask)
            val_accuracy.update(acc.item(), mask.int().sum().item())

            for data, length, ptr in zip(batch_data.cpu(), batch_lengths.cpu(),
                                         pointer_argmaxs.cpu()):
                hull_overlaps.append(calculate_hull_overlap(data, length, ptr))
        print(f'BEST Val\tLoss: {val_loss.avg:.6f} '
              f'\tAccuracy: {val_accuracy.avg:3.4%} '
              # f'\tSeq_accuracy: {Seq_accuracy.avg:3.1%} '
              f'\tOverlap: {np.mean(hull_overlaps):3.4%}')