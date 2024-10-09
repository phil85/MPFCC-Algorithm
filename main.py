# © 2023, Universität Bern, Chair of Quantitative Methods, Vanessa Tran, Manuel Kammermann, Philipp Baumann

import pandas as pd
import numpy as np
from mpfcc_algorithm import mpfcc
import matplotlib.pyplot as plt

# Read data of illustrative example
df = pd.read_csv('illustrative_example.csv')

# Extract features and colors
X = df.values[:, 1:-1]
colors = df.values[:, -1].astype(int)

# Define parameters
number_of_clusters = 3
max_cardinality = 11
min_balance = 1

# Run MPFCC-Algorithm
labels = mpfcc(X, colors, number_of_clusters, max_cardinality, min_balance,
               random_state=2, mpfcc_time_limit=300)

# Visualize resulting partition
centers = np.unique(labels)
plt.scatter(X[:, 0], X[:, 1], c=np.array(['red', 'blue'])[colors], s=30, zorder=10)
for i in range(X.shape[0]):
    plt.plot([X[i, 0], X[labels[i], 0]], [X[i, 1], X[labels[i], 1]],
             color='black', linewidth=0.8, zorder=-1, alpha=0.2)
plt.show()
