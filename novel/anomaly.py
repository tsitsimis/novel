import numpy as np
import cvxopt
import utils
import kernels


class OneClassSVM:
    def __init__(self, kernel=kernels.linear_kernel(), C=1):
        self.kernel = kernel
        self.C = C

        self.lagr_multipliers = None
        self.support_vectors = None

        self.quad_term = None
        self.radius_sqr = None

    def fit(self, X):
        n_samples, n_features = np.shape(X)
        kernel_matrix = utils.kernel_matrix(self.kernel, X)

        # Define the quadratic optimization problem
        P = cvxopt.matrix(2 * kernel_matrix, tc='d')
        q = cvxopt.matrix(-1 * kernel_matrix[np.diag_indices_from(kernel_matrix)])
        A = cvxopt.matrix(1, (1, n_samples), tc='d')
        b = cvxopt.matrix(1, tc='d')

        if not self.C:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G_max = np.identity(n_samples) * -1
            G_min = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(n_samples))
            h_min = cvxopt.matrix(np.ones(n_samples) * self.C)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))

        # Solve the quadratic optimization problem using cvxopt
        cvxopt.solvers.options['show_progress'] = False
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        lagr_mult = np.ravel(minimization['x'])

        idx = lagr_mult > 1e-7  # indexes of non-zero lagrange multipliers
        self.lagr_multipliers = lagr_mult[idx]  # corresponding lagrange multipliers
        self.support_vectors = X[idx]  # Get the samples that will act as support vectors

        # Calculate radius squared with first support vector
        self.quad_term = 0
        for i in range(len(self.lagr_multipliers)):
            for j in range(len(self.lagr_multipliers)):
                self.quad_term += self.lagr_multipliers[i] * self.lagr_multipliers[j] *\
                        self.kernel(self.support_vectors[i], self.support_vectors[j])

        self.radius_sqr = self.kernel(self.support_vectors[0], self.support_vectors[0])
        for i in range(len(self.lagr_multipliers)):
            self.radius_sqr -= 2 * self.lagr_multipliers[i] * \
                               self.kernel(self.support_vectors[i], self.support_vectors[0])
        self.radius_sqr += self.quad_term

    def predict(self, X):
        n_samples = X.shape[0]

        y_pred = np.zeros(n_samples)
        # Iterate through list of samples and make predictions
        for i in range(n_samples):
            sample = X[i, :]
            prediction = self.kernel(sample, sample)
            for j in range(len(self.lagr_multipliers)):
                prediction -= 2 * self.lagr_multipliers[j] * self.kernel(self.support_vectors[j], sample)
            prediction += self.quad_term
            y_pred[i] = prediction <= self.radius_sqr
        return y_pred
