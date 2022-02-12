import numpy as np
from scipy.optimize import Bounds, BFGS
from scipy.optimize import LinearConstraint, minimize


class MySVM:
    ZERO = 1e-7

    isTrained = False
    w_opt = -1
    b_opt = -1

    def calculate_lagrange_dual(self, alpha, x, t):
        """
        Lagrangian function L is given below
        L = ∑ αi - 0.5 * (∑ ∑ αi * αk * ti * tk * (xi.xk)

        xi and xk are vectors. All the remaining are scalars.

        Since the dual problem is to maximize L, we return -L because minimizing(-L) is same as maximizing L
        In the optimize function we are using minimize instead of maximize.

        :param alpha:
        :param x: input vector
        :param t: target

        :return: -L
        """

        alpha = alpha.reshape(alpha.shape[0], 1)
        alpha_t = np.multiply(alpha, t)
        alpha_tx = np.multiply(alpha_t.T, x.T)
        result = 0.5 * np.sum(np.dot(alpha_tx.T, alpha_tx)) - np.sum(alpha)
        return result

    def optimize_alpha(self, x, t, C):
        """
        The dual optimization problem is:

        maximize L(w, b, alpha)

        Constraints:
        αi >= 0
        ∑ αi*ti = 0

        :param x: Input vector
        :param t: target
        :param C: User defined regularization penalty

        :return: alpha
        """
        m, n = x.shape
        np.random.seed(1)
        # Initialize alphas to random values
        alpha_0 = np.random.rand(m) * C
        # Define the constraint
        linear_constraint = LinearConstraint(t, [0], [0])
        # Define the bounds
        bounds_alpha = Bounds(np.zeros(m), np.full(m, C))
        t = t.reshape(t.shape[0], 1)
        # Find the optimal value of alpha
        result = minimize(self.calculate_lagrange_dual, alpha_0, args=(x, t), method='trust-constr',
                          hess=BFGS(), constraints=[linear_constraint],
                          bounds=bounds_alpha)
        # The optimized value of alpha lies in result.x
        alpha = result.x
        return alpha

    def calculate_w_optimum(self, alpha, t, x):
        """
        w* = ∑ αi*ti*xi   (using closed form)

        :param t: target
        :param x: input
        :return: w*
        """
        m = len(x)
        # Get all support vectors
        w = np.zeros(x.shape[1])
        for i in range(m):
            w = w + alpha[i] * t[i] * x[i, :]
        return w

    def calculate_b_optimum(self, alpha, t, x, w):
        """
        b* = average(t - (w* . x))

        :param alpha: alpha
        :param t: target
        :param x: input
        :param w: optimum w
        :return: b*
        """
        ind_sv = np.where((alpha > self.ZERO))[0]
        w0 = 0.0
        for s in ind_sv:
            w0 = w0 + t[s] - np.dot(x[s, :], w)
        # Take the average
        w0 = w0 / len(ind_sv)
        return w0

    def reset(self):
        self.isTrained = False

    def train(self, x, t, C):
        """
        Calculate w_optimum and b_optimum for the SVM

        :param x: Input vectors
        :param t: target
        :param C: User defined regularization penalty
        :return:
        """
        self.isTrained = False
        alpha = self.optimize_alpha(x, t, C)
        self.w_opt = self.calculate_w_optimum(alpha, t, x)
        self.b_opt = self.calculate_b_optimum(alpha, t, x, self.w_opt)
        self.isTrained = True

    def classify(self, x_test):
        """
        calculate : (w.x) + b -> if +ve : target class = 1 otherwise -1

        :param x_test: Test points
        :return: predicted class of the test points
        """

        predicted_labels = np.sum(x_test * self.w_opt, axis=1) + self.b_opt
        predicted_labels = np.sign(predicted_labels)
        # Assign a label arbitrarily a +1 if it is zero
        predicted_labels[predicted_labels == 0] = 1
        return predicted_labels

    def accuracy(self, labels, predictions):
        N = len(labels)
        return (labels == predictions).sum() / N * 100


data = np.array([[0, 3], [-1, 0], [1, 2], [2, 1], [3, 3], [0, 0], [-1, -1], [-3, 1], [3, 1]])
labels = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1])
svm = MySVM()
svm.train(data, labels, 100)
predictions = svm.classify(data)
print(svm.accuracy(predictions, labels))