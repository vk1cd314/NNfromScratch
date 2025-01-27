import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

np.random.seed(69)

def get_data(N=300):
    X = np.random.randn(N, 2)
    T = np.where(X[:, 0]**2 + X[:, 1]**2 < 1.0, 0, 1)
    Y = np.zeros((N, 1))
    Y[T == 1] = 1
    return X, Y

class NN:
    def __init__(self, architecture):
        self.activations = []
        self.params_values = {}
        self.layers = len(architecture)
        self.grads_momentum = {}
        
        for i, layer in enumerate(architecture):
            input_size = layer["input_dim"]
            output_size = layer["output_dim"]
            activation = layer["activation"]
            
            self.activations.append(activation)
            self.params_values[f"W{i}"] = np.random.randn(output_size, input_size) / np.sqrt(input_size)
            self.params_values[f"b{i}"] = np.zeros((1, output_size))
            self.grads_momentum[f"W{i}"] = np.zeros_like(self.params_values[f"W{i}"])
            self.grads_momentum[f"b{i}"] = np.zeros_like(self.params_values[f"b{i}"])

        self.reset()

    def reset(self):
        self.cache = {}
        self.grads = {}

    def relu(self, x): return np.maximum(0, x)
    def sigmoid(self, x): return 1 / (1 + np.exp(-x))
    
    def drelu(self, dA, Z): return dA * (Z > 0).astype(float)
    def dsigmoid(self, dA, Z): 
        s = self.sigmoid(Z)
        return dA * s * (1 - s)
    
    def bce(self, yhat, y):
        eps = 1e-8
        return -np.mean(y*np.log(yhat+eps) + (1-y)*np.log(1-yhat+eps))
    
    def dbce(self, yhat, y):
        eps = 1e-8
        return -(y/(yhat+eps) - (1-y)/(1-yhat+eps))
    
    def single_forward(self, A_prev, W, b, activation):
        Z = A_prev @ W.T + b
        A = getattr(self, activation)(Z)
        return A, Z

    def forward(self, X):
        A_curr = X
        for i in range(self.layers):
            A_prev = A_curr
            W = self.params_values[f"W{i}"]
            b = self.params_values[f"b{i}"]
            activation = self.activations[i]
            
            A_curr, Z_curr = self.single_forward(A_prev, W, b, activation)
            self.cache[f"Z{i}"] = Z_curr
            self.cache[f"A{i}"] = A_prev
            
        return A_curr

    def single_backward(self, dA_curr, W, Z_curr, A_prev, activation):
        m = A_prev.shape[0]
        dZ = getattr(self, f"d{activation}")(dA_curr, Z_curr)
        dW = (dZ.T @ A_prev) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        dA_prev = dZ @ W
        
        return dA_prev, dW, db

    def backward(self, yhat, y):
        dA_prev = self.dbce(yhat, y)
        
        for i in reversed(range(self.layers)):
            W = self.params_values[f"W{i}"]
            Z_curr = self.cache[f"Z{i}"]
            A_prev = self.cache[f"A{i}"]
            activation = self.activations[i]
            
            dA_prev, dW, db = self.single_backward(dA_prev, W, Z_curr, A_prev, activation)
            self.grads[f"W{i}"] = dW
            self.grads[f"b{i}"] = db

    def accuracy(self, yhat, y): return np.mean((yhat > 0.5) == y)
    
    def update_params(self, weight_decay, momentum, lr):
        for i in range(self.layers):
            dW = self.grads[f"W{i}"] + weight_decay * self.params_values[f"W{i}"]
            db = self.grads[f"b{i}"] + weight_decay * self.params_values[f"b{i}"]
            
            self.grads_momentum[f"W{i}"] = momentum * self.grads_momentum[f"W{i}"] + (1 - momentum) * dW
            self.grads_momentum[f"b{i}"] = momentum * self.grads_momentum[f"b{i}"] + (1 - momentum) * db
            
            self.params_values[f"W{i}"] -= lr * self.grads_momentum[f"W{i}"]
            self.params_values[f"b{i}"] -= lr * self.grads_momentum[f"b{i}"]
        
        self.reset()

    def train(self, X, Y, lr=0.005, epochs=1000, momentum=0.9, weight_decay=0.0001):
        losses, accuracies = [], []
        for _ in range(epochs):
            yhat = self.forward(X)
            loss = self.bce(yhat, Y)
            acc = self.accuracy(yhat, Y)
            
            losses.append(loss)
            accuracies.append(acc)
            
            self.backward(yhat, Y)
            self.update_params(weight_decay, momentum, lr)
            
        return losses, accuracies

nn_architecture = [
    {"input_dim": 2, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 3, "activation": "relu"},
    {"input_dim": 3, "output_dim": 10, "activation": "relu"},
    {"input_dim": 10, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
]

training_sizes = [50, 100, 200, 300, 500, 1000]
test_sizes = [50, 100, 200, 300, 500, 1000]
results = np.zeros((len(training_sizes), len(test_sizes)))

test_sets = {size: get_data(size) for size in test_sizes}

for train_idx, train_size in enumerate(training_sizes):
    X_train, Y_train = get_data(train_size)
    model = NN(nn_architecture)
    model.train(X_train, Y_train, lr=0.005, epochs=1000)
    
    for test_idx, test_size in enumerate(test_sizes):
        X_test, Y_test = test_sets[test_size]
        yhat = model.forward(X_test)
        acc = model.accuracy(yhat, Y_test)
        results[train_idx, test_idx] = acc

table = []
headers = ["Train\\Test"] + [str(s) for s in test_sizes]
for train_size, row in zip(training_sizes, results):
    formatted_row = [f"{train_size}"] + [f"{acc:.2%}" for acc in row]
    table.append(formatted_row)

print(tabulate(table, headers=headers, tablefmt="grid"))

plt.figure(figsize=(10, 8))
plt.imshow(results, cmap='viridis', origin='upper')
plt.colorbar(label='Accuracy')
plt.xticks(np.arange(len(test_sizes)), test_sizes)
plt.yticks(np.arange(len(training_sizes)), training_sizes)
plt.xlabel("Test Size")
plt.ylabel("Training Size")
plt.title("Generalization Across Sizes")
plt.savefig('size_generalization.png')
plt.show()
