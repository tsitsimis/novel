import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs, make_moons
from novel import OneClassSVM as OneSVM
import novel.kernels as kernels


# data
# X, y = make_blobs(n_samples=100, centers=2, random_state=42)
X, y = make_moons(n_samples=300, noise=0.1, random_state=42)
X = X[y == 1]

# classifier
kernel = kernels.rq_kernel(1.5)
clf = OneSVM(kernel, C=1)
clf.fit(X)

# decision boundary
f1 = 0
f2 = 1
h = 0.05
x_min, x_max = X[:, f1].min() - 1.0, X[:, f1].max() + 1.0
y_min, y_max = X[:, f2].min() - 1.0, X[:, f2].max() + 1.0
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cm_bright = ListedColormap(['#FF0000', '#0000FF'])
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c='b', edgecolors='k')
plt.scatter(clf.support_vectors[:, 0], clf.support_vectors[:, 1], facecolors='none', edgecolors='w')

plt.show()
