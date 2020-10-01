import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestCentroid
import matplotlib.pyplot as plt

X, y = make_blobs(centers=2, cluster_std=2, random_state=0, n_samples=50)


def plot_2d_classification(classifier, X, fill=False, ax=None, eps=None, alpha=1):
    # multiclass
    if eps is None:
        eps = X.std() / 2.

    if ax is None:
        ax = plt.gca()

    x_min, x_max = X[:, 0].min() - eps, X[:, 0].max() + eps
    y_min, y_max = X[:, 1].min() - eps, X[:, 1].max() + eps
    xx = np.linspace(x_min, x_max, 1000)
    yy = np.linspace(y_min, y_max, 1000)

    X1, X2 = np.meshgrid(xx, yy)
    X_grid = np.c_[X1.ravel(), X2.ravel()]
    decision_values = classifier.predict(X_grid)
    ax.imshow(decision_values.reshape(X1.shape), extent=(x_min, x_max,
                                                         y_min, y_max),
              aspect='auto', origin='lower', alpha=alpha)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())


plt.figure()

nc = NearestCentroid()
nc.fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
plot_2d_classification(nc, X, alpha=.5)
plt.gca().set_aspect("equal")
plt.scatter(nc.centroids_[:, 0], nc.centroids_[:, 1], c=['b', 'r'], s=100, marker='x')
plt.savefig("images/nearest_centroid_boundary.png", bbox_inches='tight')
