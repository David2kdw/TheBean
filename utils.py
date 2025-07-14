import numpy as np

def manhattan_dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_dist_to_nearest(items, px, py, metric="manhattan"):
    if not items:
        return 0.0          # no items → distance zero by convention
    if metric == "euclid":
        dists = [np.hypot(px - x, py - y) for x, y in items]
    else:
        dists = [manhattan_dist((px, py), (x, y)) for x, y in items]
    return min(dists)