# © 2023, Universität Bern, Chair of Quantitative Methods, Vanessa Tran, Manuel Kammermann, Philipp Baumann

import gurobipy as gb
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import kmeans_plusplus
import time


def rename_labels(labels, center_index):
    """ Renames labels according to correct index

    Args:
        labels (np.array): cluster labels for objects
        center_index (np.array): index of cluster centers in initial solution

    Returns:
        labels (np.array): re-indexed cluster labels for objects

    """

    # Create dictionary
    transdict = dict()

    # Translate index of cluster center from initial solution to index of cluster center in data set
    for old, new in zip(np.unique(labels), center_index):
        transdict[old] = new

    # re-labeling according to transdict
    # labels should correspond to correct indices of centers in the data set
    labels = np.asarray(list(transdict.values()))[labels]

    return labels


def get_total_distance(X, centers, labels):
    """Computes total distance between objects and cluster centers

    Args:
        X (np.array): feature vectors of objects
        centers (np.array): current positions of cluster centers
        labels (np.array): current cluster assignments of objects

    Returns:
        dist (float): total distance
        distance_time (float): time used to compute distance

    """

    # Set timer
    distance_time = time.time()

    # Determine clustering cost for different objective functions
    dist = np.sqrt(((X - centers[labels, :]) ** 2).sum(axis=1)).sum()
    # dist = sum(np.linalg.norm(X - centers[labels, :], axis=1))

    # Determine distance computation time
    distance_time = time.time() - distance_time

    return dist, distance_time


def update_centers(X, centers, n_clusters, labels):
    """Update positions of cluster centers

    Args:
        X (np.array): feature vectors of objects
        centers (np.array): current positions of cluster centers
        n_clusters (int): predefined number of clusters
        labels (np.array): current cluster assignments of objects

    Returns:
        centers (np.array): the updated positions of cluster centers

    """

    # Update timer
    update_time = time.time()
    # Initialize center_index
    center_index = list()

    # Create array with indices
    index = np.arange(len(X))

    # iterate over clusters
    for i in range(n_clusters):

        # Compute medoid of each cluster (k-medoid): If number of objects exceed 50,000 then randomly sample
        # 50,000 objects from the cluster
        # Sample 50'000 random objects if size of cluster is too big
        np.random.seed(0)
        x_sample = X[labels == i, :]
        idx = index[labels == i]

        # Compute pairwise euclidean distance of each data point in cluster
        distances = pd.DataFrame(cdist(x_sample, x_sample, metric='euclidean'), index=idx, columns=idx)

        centers[i] = x_sample[distances.sum(axis=0).argmin()]
        center_index.append(distances.sum(axis=0).idxmin())

    # Determine update step time
    update_time = time.time() - update_time

    return centers, center_index, update_time


def assign_objects(X, centers, colors, balance, cardinality):
    """Assigns objects to clusters

    Args:
        X (np.array): feature vectors of objects
        centers (np.array): current positions of cluster centers
        colors (np.array): colors of objects (0=red; 1=blue)
        balance (float): balance
        cardinality (int): max. cluster cardinality

    Returns:
        labels (np.array): cluster labels for objects

    """

    # Compute model input
    n = X.shape[0]
    k = centers.shape[0]
    distances = cdist(X, centers)
    assignments = {(i, j): distances[i, j] for i in range(n) for j in range(k)}
    red = np.where(colors == 0)[0]
    blue = np.where(colors == 1)[0]

    # Create model
    m = gb.Model()

    # Add binary decision variables
    y = m.addVars(assignments, obj=assignments, vtype=gb.GRB.BINARY)

    # Add constraints
    m.addConstrs(y.sum(i, '*') == 1 for i in range(n))
    m.addConstrs(y.sum('*', j) >= 1 for j in range(k))

    # add cardinality constraint
    m.addConstrs(gb.quicksum(y[i, j] for i in range(n)) <= cardinality for j in range(k))

    m.addConstrs(gb.quicksum(y[i, j] for i in red) >= balance * gb.quicksum(y[i, j] for i in blue) for j in range(k))
    m.addConstrs(gb.quicksum(y[i, j] for i in blue) >= balance * gb.quicksum(y[i, j] for i in red) for j in range(k))

    # Determine optimal solution
    m.setParam('Outputflag', 0)
    # m.setParam('TimeLimit', 3600)
    m.optimize()

    # Get labels from optimal assignment
    if m.status != gb.GRB.INFEASIBLE:
        try:
            labels = np.array([j for i, j in y.keys() if y[i, j].X > 0.5])
        except:
            labels = []
    else:
        labels = []

    run_time = m.Runtime

    return labels, run_time, m.status


def get_balance(labels, colors):
    """ Computes balance of the clustering

    Args:
        labels (np.array): current cluster assignments of objects
        colors (np.array): colors of objects (0=red; 1=blue)

    Returns:
        min_ (float): achieved balance

    """

    # Set timer
    bal_time = time.time()

    # Initialize balance with the highest possible value of 1
    min_ = 1

    # Determine best achieved balance
    for cluster in np.unique(labels):

        # Determine number of red and blue objects within the cluster
        r = sum(colors[labels == cluster])
        b = len(colors[labels == cluster]) - r

        # Compute balance
        if r == 0 or b == 0:
            min_ = 0

        else:
            min_ = min(min_, r / b, b / r)

    return min_, time.time() - bal_time


def mpfcc(X, colors, n_clusters, cardinality, balance, random_state, mpfcc_time_limit):
    """Finds partition of X subject to balance and cardinality constraint

    Args:
        X (np.array): feature vectors of objects
        n_clusters (int): predefined number of clusters
        colors (np.array): colors of objects (0=red; 1=blue)
        balance (float): balance
        cardinality (int): max. cluster cardinality
        random_state (int, RandomState instance): random state
        mpfcc_time_limit (float): time limit for construction heuristic

    Returns:
        best_labels (np.array): cluster labels of objects
        best_total_distance (float): minimal distance (objective function value)
        total_time (float): total running time
        best_total_balance (float): achieved balance in best solution

    """

    # Initialize start time and iteration step
    start_time = time.time()

    # Choose initial cluster using the k-means++ algorithm
    centers, indices = kmeans_plusplus(X, n_clusters=n_clusters, random_state=random_state, n_local_trials=1000)

    # Assign objects
    labels, run_time, status = assign_objects(X, centers, colors, balance, cardinality)

    if status != gb.GRB.INFEASIBLE and len(labels) > 0:

        # Initialize best labels
        best_labels = labels

        # Update centers
        centers, center_index, update_time = update_centers(X, centers, n_clusters, labels)

        # Compute total distance
        best_total_distance, distance_time = get_total_distance(X, centers, labels)

        while time.time() - start_time < mpfcc_time_limit:

            labels, run_time, status = assign_objects(X, centers, colors, balance, cardinality)

            if len(labels) > 0:
                # Update centers
                centers, center_index, update_time = update_centers(X, centers, n_clusters, labels)

                # Compute total distance
                total_distance, distance_time = get_total_distance(X, centers, labels)

                # Compute balance
                balance, balance_time = get_balance(labels, colors)

                # Check stopping criterion
                if total_distance >= best_total_distance:
                    break

                else:
                    best_labels = labels
                    best_total_distance = total_distance
            else:
                break

        total_time = time.time() - start_time
        print('kmedoid cost: ' + str(best_total_distance))
        print("Total running time: " + str(total_time))

    elif status != gb.GRB.TIME_LIMIT:
        center_index = []
        best_labels = []
        best_total_distance = 'infeasible'
        total_time = time.time() - start_time
        print('kmedoid cost: ' + str(best_total_distance))
        print("Total running time: " + str(total_time))

    else:
        center_index = []
        best_labels = []
        best_total_distance = 'time limit'
        print('kmedoid cost: ' + str(best_total_distance))
        print("Total running time: " + 'time limit reached (no feasible solution within timelimit)')

    labels = rename_labels(best_labels, center_index)

    return labels
