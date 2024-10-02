import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import warnings
warnings.filterwarnings('ignore')

# logistic activation function: Sigmoid function used for non-linear activations
logistic = lambda z: 1. / (1 + np.exp(-z))

# definition of the MLP class:
class MLP:
    def __init__(self, M=64):
        self.M = M

    # The 'fit' method should be indented inside the class
    def fit(self, x, y, optimizer):
        N, D = x.shape
        def gradient(x, y, params):
            v, w = params
            z = logistic(np.dot(x, v))  # N x M, x - input data, v - weights
            yh = logistic(np.dot(z, w))  # N, z - hidden layer output
            dy = yh - y  # N, error in the network predictions (predicted values - true labels)
            dw = np.dot(z.T, dy) / N  # M, gradient of loss with respect to weights w
            dz = np.outer(dy, w)  # N x M, dz calculates the error propagated back from the output layer
            dv = np.dot(x.T, dz * z * (1 - z)) / N  # D x M, gradient for input-to-hidden weights
            dparams = [dv, dw]  # Gradients are stored in a list and returned to update the model's weights
            return dparams

        w = np.random.randn(self.M) * 0.01  # Initialize weights for the hidden-to-output layer
        v = np.random.randn(D, self.M) * 0.01  # Initialize weights for the input-to-hidden layer
        params0 = [v, w]  # Initial weights
        # We initialize weights v and w with small random values (multiplied by 0.01). 
        # These are starting points for the optimization process.
        self.params = optimizer.run(gradient, x, y, params0)  # Call the optimizer to minimize the error
        # The trained parameters are stored in self.params
        return self

    # The 'predict' method should also be indented inside the class
    def predict(self, x):
        v, w = self.params  # Use trained parameters
        z = logistic(np.dot(x, v))  # Compute hidden layer output
        yh = logistic(np.dot(z, w))  # Compute final output
        return yh  # Return the predicted output (probabilities)

# Define the GradientDescent class for optimization:
class GradientDescent:
    def __init__(self, learning_rate=0.001, max_iters=1e4, epsilon=1e-8):
        self.learning_rate = learning_rate  # Step size for weight updates
        self.max_iters = max_iters  # Maximum number of iterations
        self.epsilon = epsilon  # Stopping criteria based on gradient size
        # The learning rate controls the size of each weight update, while max_iters limits the iterations,
        # and epsilon stops the optimizer when the gradient is small enough.

    def run(self, gradient_fn, x, y, params):
        norms = np.array([np.inf])  # Initialize gradient norm to a large value
        t = 1 
        while np.any(norms > self.epsilon) and t < self.max_iters:
            grad = gradient_fn(x, y, params)  # Compute gradients
            for p in range(len(params)):
                params[p] -= self.learning_rate * grad[p]  # Update each parameter
            t += 1
            norms = np.array([np.linalg.norm(g) for g in grad])  # Calculate gradient norms
        return params  # Return the optimized parameters

# This function repeatedly calculates gradients and updates the weights using gradient descent
# until the gradients become small enough or max_iters is reached.

# Load the dataset and train the model:
dataset = datasets.load_iris()  # Load the Iris dataset
x, y = dataset['data'][:, [1, 2]], dataset['target']  # Use sepal width and length as features
y = y == 1  # Convert target to binary classification (class 1 vs others)

# We load the Iris dataset and select two features for simplicity. The target is converted to binary 
# to classify whether a sample belongs to class 1 or not.

model = MLP(M=32)  # Create an MLP instance with 32 hidden neurons
optimizer = GradientDescent(learning_rate=0.1, max_iters=20000)  # Initialize optimizer
yh = model.fit(x, y, optimizer).predict(x)  # Train the model and predict on the same data

# The model is trained using gradient descent. yh contains the predictions after training.

# Plot the decision boundary:
x0v = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 200)  # Create a grid of values for plotting
x1v = np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), 200)
x0, x1 = np.meshgrid(x0v, x1v)
x_all = np.vstack((x0.ravel(), x1.ravel())).T  # Combine into a grid of points

yh_all = model.predict(x_all) > 0.5  # Predict class labels for each grid point

# This creates a grid of points covering the input feature space. The model predicts the class for each point,
# which helps in visualizing the decision boundary.

plt.scatter(x[:, 0], x[:, 1], c=y, marker='o', alpha=1)  # Scatter plot of actual data points
plt.ylabel('sepal length')
plt.xlabel('sepal width')
plt.title('Decision Boundary of the MLP')
plt.show()
