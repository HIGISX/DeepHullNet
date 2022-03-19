from scipy.spatial import ConvexHull
from concavehull import concaveHull
import numpy as np
import json
import datetime
import os
TOKENS = {
    '<sos>': 0,
    '<eos>': 1
}
def save_data(data, cfg, name=None, data_dir='./dataset'):
    """
    Data will be saved in json format
    save the configuration and data both
    in the json file.

    Args:
        data: data generated with optimal solutions
        cfg: configuration loaded from file
    Return:
        file_name: absolute path for the data file
    """
    s_data = {'cfg':cfg, 'data': data}
    if not name:
        now = datetime.datetime.now()
        name = 'dataset-%02d-%02d-%2d-%2d-%2d.json' % (now.day, now.month, now.hour, now.minute, now.second)

    if os.path.exists(data_dir):
        file_name = os.path.join(data_dir, name)
    else:
        os.makedirs(data_dir)

    file_name = os.path.join(data_dir, name)

    with open(file_name, 'w') as fp:
        fp.write(json.dumps(s_data, indent=3))
    return file_name


def generate_random_points(n, sort_method: str = 'lex'):
    """
    Randomly sample n sorted uniformly distributed 2D points from [0.0, 1.0).

    Args:
    n: Number of x,y points to generate.
    sort_method: Method to sort points. The following methods are supported:
      lex: Sort in ascending lexicographic order starting from y.
      mag: Sort from least to greatest magnitude (or distance from origin).
    Outputs:
    Shape (n, 2) sorted numpy array of x,y coordinates.
    """

    points = np.random.random(size=[n, 2])
    if sort_method == 'lex':
        points = points[np.lexsort(([points[..., ax] for ax in range(points.shape[-1])]))]
    elif sort_method == 'mag':
        points = points[np.argsort(np.mean(points, axis=-1))]
    else:
        raise ValueError(f'{sort_method} is not a valid option for sort_method.')
    return points

def cyclic_permute(l, idx):
  """
  Permute a list such that l[idx] becomes l[0] while preserving order.

  Args:
    l: List to permute.
    idx: Index to the element in l that should appear at position 0.
  Outputs:
    Cyclically permuted list of length len(l).
  """
  return l[idx:] + l[:idx]

def gen_targets(points, concave_hull):
  list_points = points.tolist()
  targets = []
  for i in range(len(concave_hull)):
    point = concave_hull[i].tolist()
    target = list_points.index(point)
    targets.append(target)
  return targets

def generate_convex_hull(min_nodes, max_nodes, num_samples):
    data = []
    total_points = []
    total_targets = []
    n_points = np.random.randint(low=min_nodes, high=max_nodes + 1, size=num_samples)
    for i in n_points:
        points = generate_random_points(i).tolist()
        targets = ConvexHull(points).vertices.tolist()
        targets = cyclic_permute(targets, np.argmin(targets))
        total_points.append(points)
        total_targets.append(targets)
    data.append({
        'points': total_points,
        'targets': total_targets,
    })
    return data

def generate_concave_hull(min_nodes, max_nodes, num_samples):
    data = []
    total_points = []
    total_targets = []
    n_points = np.random.randint(low=min_nodes, high=max_nodes + 1, size=num_samples)
    for i in n_points:
        points = generate_random_points(i)
        k = 3
        concave_hull = concaveHull(points, k)
        targets = gen_targets(points, concave_hull)
        # targets = [i for i, val in enumerate(concave_hull.indices) if val]
        targets = cyclic_permute(targets, np.argmin(targets))

        total_points.append(points.tolist())
        total_targets.append(targets)
    data.append({
        'points': total_points,
        'targets': total_targets,
    })
    return data
