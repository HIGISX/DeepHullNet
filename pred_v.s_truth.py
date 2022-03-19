from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from solver import generate_convex_hull, save_data, cyclic_permute, generate_concave_hull, gen_targets
from process import Scatter2DDataset
from torch.utils.data import DataLoader
from model import DeepHullNet
import torch
import numpy as np
from compare_plot import display_points_with_hull, display_points
from cal import calculate_hull_overlap
from concavehull import concaveHull
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

TOKENS = {
    '<sos>': 0,
    '<eos>': 1
}

if __name__ == '__main__':

    problem = 'convex'
    idx = 0
    min_nodes = 200
    max_nodes = 200
    n = 5
    use_cuda = True
    plot = True

    cfg = {'min_nodes': min_nodes, 'max_nodes': max_nodes}

    if problem == 'convex':
        eval_dataset = generate_convex_hull(min_nodes, max_nodes, num_samples=n)
        eval_file_name = save_data(eval_dataset, cfg, name='eval_dataset.json',
                                    data_dir=f'./dataset/eval/convex/{min_nodes}-{max_nodes}/')
        print("%d convex eval samples have been generated." % (n))
        eval_dataset = Scatter2DDataset(file_name=f'./dataset/eval/convex/{min_nodes}-{max_nodes}/eval_dataset.json')
        eval_loader = DataLoader(
            eval_dataset, batch_size=1, shuffle=True,
            num_workers=0, pin_memory=False)
        device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        model = DeepHullNet(c_inputs=5, c_embed=16, n_heads=4,
                          n_layers=3, dropout=0.0, c_hidden=16).to(device)

        checkpoint_file = f'./checkpoint/convex/5-50/Transform/best_params.pkl'
        model.load_state_dict(torch.load(checkpoint_file)['model_state'])

        batch_data, batch_labels, batch_lengths = next(iter(eval_loader))
        log_pointer_scores, pointer_argmaxs = model(batch_data.to(device),
                                                    batch_lengths.to(device), batch_labels=batch_labels)

        pred_hull_idxs = pointer_argmaxs[idx].cpu()
        pred_hull_idxs = pred_hull_idxs[pred_hull_idxs > 1] - 2
        points = batch_data[idx, 2:batch_lengths[idx], :2]
        true_hull_idxs = ConvexHull(points).vertices.tolist()
        true_hull_idxs = cyclic_permute(true_hull_idxs, np.argmin(true_hull_idxs))

        print(f'Predicted: {pred_hull_idxs.tolist()}')
        print(f'True:      {true_hull_idxs}')

        if plot == True:
            plt.rcParams['figure.figsize'] = (10, 6)
            plt.subplot(1, 2, 1)
            true_hull_idxs = ConvexHull(points).vertices.tolist()
            display_points(points)
            display_points_with_hull(points, true_hull_idxs)
            _ = plt.title('Convex Hull')

            plt.subplot(1, 2, 2)
            display_points(points)
            display_points_with_hull(points, pred_hull_idxs)
            _ = plt.title('DeepHullNet Convex Hull')

            overlap = calculate_hull_overlap(batch_data[idx], batch_lengths[idx],
                                             pointer_argmaxs[idx])
            print(f'Hull overlap: {overlap:3.2%}')
            plt.show()

    elif problem == 'concave':
        eval_dataset = generate_concave_hull(min_nodes, max_nodes, num_samples=n)
        eval_file_name = save_data(eval_dataset, cfg, name='eval_dataset.json',
                                    data_dir=f'./dataset/eval/concave/{min_nodes}-{max_nodes}/')
        print("%d concave eval samples have been generated." % (n))
        eval_dataset = Scatter2DDataset(file_name=f'./dataset/eval/concave/{min_nodes}-{max_nodes}/eval_dataset.json')
        eval_loader = DataLoader(
            eval_dataset, batch_size=1, shuffle=True,
            num_workers=0, pin_memory=False)
        device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        model = DeepHullNet(c_inputs=5, c_embed=16, n_heads=4,
                          n_layers=3, dropout=0.0, c_hidden=16).to(device)

        checkpoint_file = f'./checkpoint/concave/5-50/Transform/best_params.pkl'
        model.load_state_dict(torch.load(checkpoint_file)['model_state'])

        batch_data, batch_labels, batch_lengths = next(iter(eval_loader))
        log_pointer_scores, pointer_argmaxs = model(batch_data.to(device),
                                                    batch_lengths.to(device), batch_labels=batch_labels)

        pred_hull_idxs = pointer_argmaxs[idx].cpu()
        pred_hull_idxs = pred_hull_idxs[pred_hull_idxs > 1] - 2
        points = np.array(batch_data[idx, 2:batch_lengths[idx], :2])
        k = 3
        concave_hull = concaveHull(points, k)
        true_hull_idxs = gen_targets(points, concave_hull)
        true_hull_idxs = cyclic_permute(true_hull_idxs, np.argmin(true_hull_idxs))

        print(f'Predicted: {pred_hull_idxs.tolist()}')
        print(f'True:      {true_hull_idxs}')

        if plot == True:
            plt.rcParams['figure.figsize'] = (10, 6)
            plt.subplot(1, 2, 1)
            # true_hull_idxs = ConvexHull(points).vertices.tolist()
            display_points(points)
            display_points_with_hull(points, true_hull_idxs)
            _ = plt.title('Concave Hull')

            plt.subplot(1, 2, 2)
            display_points(points)
            display_points_with_hull(points, pred_hull_idxs)
            _ = plt.title('DeepHullNet Concave Hull')

            # plt.subplot(1, 3, 3)
            # true_hull_idxs = ConvexHull(points).vertices.tolist()
            # display_points(points)
            # display_points_with_hull(points, true_hull_idxs)
            # _ = plt.title('Shapely Concave Hull')

            overlap = calculate_hull_overlap(batch_data[idx], batch_lengths[idx],
                                             pointer_argmaxs[idx])
            print(f'Hull overlap: {overlap:3.2%}')
            plt.show()
    else:
        raise NotImplementedError

