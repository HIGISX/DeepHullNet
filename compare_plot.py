import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from concavehull import concaveHull
from solver import generate_random_points, gen_targets

def display_points(points):
    """
    Display a set of 2D points on a scatterplot.

    Args:

    Returns:
        object:
    points: x,y coordinate points.
    """

    y_offset = 0.025
    plt.scatter(points[:, 0], points[:, 1])
    for i, point in enumerate(points):
        plt.text(point[0], point[1] + y_offset, str(i))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(0., 1.)
        plt.ylim(0., 1.)
        plt.title(f'N: {len(points)}')
        plt.grid(True)

def display_points_with_hull(points, hull):
    """
    Display a set of 2D points with its convex hull.

    Args:
    points: x,y coordinate points.
    hull: List of indices indicating the convex hull of points.
    """

    for i in range(len(hull) - 1):
        p0 = hull[i]
        p1 = hull[i + 1]
        x = points[[p0, p1], 0]
        y = points[[p0, p1], 1]
        plt.plot(x, y, 'g')
        plt.arrow(x[0], y[0], (x[1] - x[0]) / 2., (y[1] - y[0]) / 2.,
                  shape='full', lw=0, length_includes_head=True, head_width=.025,
                  color='green')
    x = points[[p1, hull[0]], 0]
    y = points[[p1, hull[0]], 1]
    plt.arrow(x[0], y[0], (x[1] - x[0]) / 2., (y[1] - y[0]) / 2.,
            shape='full', lw=0, length_includes_head=True, head_width=.025,
            color='green')
    plt.plot(x, y, 'g')
    plt.grid(True)

if __name__ == '__main__':
    nodes = 20
    points = generate_random_points(nodes)
    display_points(points)
    convex_hull_sequence = ConvexHull(points).vertices.tolist()
    k = 3
    concave_hull = concaveHull(points, k)
    concave_hull_sequence = gen_targets(points, concave_hull)
    display_points_with_hull(points, convex_hull_sequence)
    display_points_with_hull(points, concave_hull_sequence)
    plt.show()