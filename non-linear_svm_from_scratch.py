import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import Bounds, BFGS
from scipy.optimize import LinearConstraint, minimize

"""
Implementation of Non-Linear SVM with polynomial and gaussian (rbf) kernel.
"""


class MySVM_NonLinearKernel:
    ZERO = 1e-7

    isTrained = False
    w_opt = -1
    b_opt = -1
    alpha = []
    KernelFunction = None

    def __init__(self, kernel="poly"):
        if kernel.lower() == "poly":
            self.KernelFunction = self.polynomialKernelTransform
        elif kernel.lower() == "rbf":
            self.KernelFunction = self.gaussianKernelTransform

    def lagrange_dual(self, alpha, x, t):
        """
        Lagrangian function L is given below
        L = ∑ αi - 0.5 * (∑ ∑ αi * αk * ti * tk * (f(xi).f(xk))

        xi and xk are vectors. All the remaining are scalars.

        f is a polynomial transformation to a higher dimension space. Here, we apply the
        kernel trick. Instead of computing f(xi).f(xk) we get all the required terms from (1+xi.xk)^n.

        In comparison to linear svm, the non-linear svm fits the hyperplane in the f space.

        Since the dual problem is to maximize L, we return -L because minimizing(-L) is same as maximizing L
        In the optimize function we are using minimize instead of maximize.

        :param alpha:
        :param x: input vector
        :param t: target

        :return: -L
        """

        result = 0
        ind_sv = np.where(alpha > self.ZERO)[0]
        for i in ind_sv:
            for k in ind_sv:
                result = result + alpha[i] * alpha[k] * t[i] * t[k] * self.KernelFunction(x[i, :], x[k, :])
        result = 0.5 * result - sum(alpha)
        return result

    def gaussianKernelTransform(self, X, Z):
        """
        The Kernel function for gaussian (RBF) transformation is:

        exp(−gamma * |x−z|^2)

        where gamma is a hyper-parameter. I have set it to 0.1.

        :param X:
        :param Z:
        :return:
        """
        gamma = 0.1
        return np.exp(-gamma * np.linalg.norm(X - Z) ** 2)

    def polynomialKernelTransform(self, X, Z):
        """
        The Kernel function for polynomial transformation is:
        (1 + x.z)^n

        Where n is a hyper-parameter that indicates the degree of the polynomial. We have chosen 3 here.

        :param X:
        :param Z:
        :return:
        """
        return (np.dot(X, Z) + 1) ** 3

    def optimize_alpha(self, x, t, C):
        """
        (same as linear SVM)
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
        result = minimize(self.lagrange_dual, alpha_0, args=(x, t), method='trust-constr',
                          hess=BFGS(), constraints=[linear_constraint],
                          bounds=bounds_alpha)
        # The optimized value of alpha lies in result.x
        alpha = result.x
        return alpha

    def classify_points(self, alpha, t, x, x_test):
        """
        w* = ∑ αi*ti*f(xi)
        y_test = w*.f(x_test) + b*
               = ∑ αi*ti*f(xi).f(x_test) + b*

        Here again we can apply the kernel trick instead of explicitly calculating f(xi).f(x_test)
        :param alpha:
        :param t:
        :param x:
        :param x_test:
        :return y_test:
        """
        b = self.calculate_b_optimum(alpha, t, x)
        m = len(x)
        n = len(x_test)
        y_test = []
        for j in range(n):
            p = b
            for i in range(m):
                p += alpha[i] * t[i] * self.KernelFunction(x[i, :], x_test[j, :])

            y_test.append(p)
        return np.sign(y_test)

    def calculate_b_optimum(self, alpha, t, x):
        """
        b* = average(t - (w* . f(x)))
        b* = average(t - (∑ αi*ti*f(xi).f(x)))

        We can apply the kernel trick the calculate f(xi).f(x) without explicitly transforming x to f(x)

        :param alpha: alpha
        :param t: target
        :param x: input
        :param w: optimum w
        :return: b*
        """
        ind_sv = np.where((alpha > self.ZERO))[0]
        w0 = 0.0
        m = len(x)
        for s in ind_sv:
            w = 0
            for i in range(m):
                w = w + alpha[i] * t[i] * self.KernelFunction(x[i, :], x[s, :])
            w0 = w0 + t[s] - w
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
        self.alpha = self.optimize_alpha(x, t, C)
        self.isTrained = True

    def accuracy(self, labels, predictions):
        """
        Count the percentage of correctly predicted labels of the test set

        :param labels:
        :param predictions:
        :return:
        """
        N = len(labels)
        count_corrects = 0
        for i in range(N):
            if labels[i] == predictions[i]:
                count_corrects += 1

        return count_corrects / N * 100

    def plot_x_train(self, x, labels):
        """
        Plot training data
        :param x:
        :param labels:
        :return:
        """
        sns.scatterplot(x[:, 0], x[:, 1], style=labels,
                        hue=labels, markers=['s', 'P'],
                        palette=['magenta', 'green'])

    def plot_x_test(self, x, labels):
        """
        Plot test data
        :param x:
        :param labels:
        :return:
        """
        sns.scatterplot(x[:, 0], x[:, 1], style=labels,
                        hue=labels, markers=['s', 'P'],
                        palette=['red', 'black'])

    def plotTruePolyFunction(self):
        """
        This plots ths true non-linear function which is a 5th degree polynomial. The svm's task is to approximate this function
        as much as possible. It achieves 100% accuracy on the test set with a 3rd degree polynomial transformation.
        :return:
        """

        def getTrueY(x):
            return 3 * pow(x, 5) + 4 * pow(x, 4) + 7 * pow(x, 3) - 9 * pow(x, 2) + 1

        x = []
        y = []
        start = -1
        while start <= 1:
            x.append(start)
            y.append(getTrueY(start))
            start += 0.2
        plt.plot(x, y)


def executeExamplePolynomialSVM():
    svm2 = MySVM_NonLinearKernel("poly")

    x = np.array(
        [[-0.5, 0], [-0.7, -2], [0, 1.2], [0.3, 0.9], [0.7, 3], [-0.5, 0], [0.5, 0.3], [-0.6, -1], [0, 0.5], [0.3, 0],
         [0.4, -0.5], [1, 1], [1, 2], [-0.1, -0.5], [-0.3, -0.6], [-0.4, -3]])
    labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1])

    # Plot the training data along with the true function
    svm2.plotTruePolyFunction()
    svm2.plot_x_train(x, labels)
    plt.savefig('training-data-polynomial-svm.png')
    plt.clf()

    x_test = np.array(
        [[2, 2], [-0.5, 2], [0.5, -2], [-2, 0.5], [0, 0], [0.3, 1.2], [0, 0.3], [0.1, 1.3], [0.6, 0.4], [0.6, -0.4],
         [0.7, 4], [-2, -3]])
    labels_test = np.array([-1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1])

    # Plot the testing data along with the true function
    svm2.plotTruePolyFunction()
    svm2.plot_x_test(x_test, labels_test)
    plt.savefig('testing-data-polynomial-svm.png')

    C = 100
    # SVM trains using the training data and corresponding target values in a supervised fashion.
    svm2.train(x, labels, C)

    # SVM inference uses the test data. It does not use labels_test which is the expected class of the test data samples.
    y_test = svm2.classify_points(svm2.alpha, labels, x, x_test)

    # Once prediction is obtained from SVM, we measure the accuracy using labels of the test data.
    print("Accuracy on test set of Polynomial SVM ", svm2.accuracy(y_test, labels_test))


def executeExampleRBFSVM():
    svm2 = MySVM_NonLinearKernel("RBF")

    x = np.array(
        [[2.8, 2], [3.1, 4], [-1, 4], [-0.5, 1], [1, 1], [1, 5], [1.8, 1.3], [2.9, 4.5], [1.7, 5.3], [-1.3, 1.6],
         [-1.5, 2.6], [1, 3], [0.5, 2.5], [0.8, 3.5], [1.6, 3.5], [1.5, 2.9], [0.2, 3], [0.2, 3.5], [1.7, 2.5]])
    labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1])

    # Plot the training data along with the true function
    circle = plt.Circle((1, 3), 1.732)
    fig, ax = plt.subplots()
    ax.add_patch(circle)
    svm2.plot_x_train(x, labels)
    plt.savefig('training-data-rbf-svm.png')
    plt.clf()

    x_test = np.array(
        [[0.4, 3.6], [1, 4], [1.3, 2.4], [2, 3], [0.4, 2.9], [-1, 1.4], [2.5, 5], [2.2, 1.4], [-0.9, 4.6], [1, 0.2],
         [4, 3], [-3, 1]])
    labels_test = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1])

    # Plot the testing data along with the true function
    circle = plt.Circle((1, 3), 1.732)
    fig, ax = plt.subplots()
    ax.add_patch(circle)
    svm2.plot_x_test(x_test, labels_test)
    plt.savefig('testing-data-rbf-svm.png')

    C = 100
    # SVM trains using the training data and corresponding target values in a supervised fashion.
    svm2.train(x, labels, C)

    # SVM inference uses the test data. It does not use labels_test which is the expected class of the test data samples.
    y_test = svm2.classify_points(svm2.alpha, labels, x, x_test)
    # Once prediction is obtained from SVM, we measure the accuracy using labels of the test data.
    print("Accuracy on test set of RBF SVM", svm2.accuracy(y_test, labels_test))


executeExamplePolynomialSVM()

executeExampleRBFSVM()
