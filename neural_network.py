import numpy as np

class NeuralNetwork:
    def __init__(self, layers, activation='tanh', output_activation='sigmoid', learning_rate=0.01, regularization=0.01):
        """
        Initialize a neural network capable of learning non-linear decision boundaries
        
        Parameters:
        - layers: list of integers representing the number of neurons in each layer
        - activation: activation function for hidden layers ('tanh', 'relu', 'leaky_relu')
        - output_activation: activation function for output layer ('sigmoid', 'softmax')
        - learning_rate: learning rate for gradient descent
        - regularization: L2 regularization strength
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.weights = []
        self.biases = []
        self.activation = activation
        self.output_activation = output_activation
        
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            # Xavier/Glorot initialization
            scale = np.sqrt(2.0 / (layers[i] + layers[i+1]))
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * scale)
            self.biases.append(np.zeros((1, layers[i+1])))
    
    def _activate(self, x, activation):
        """Apply activation function"""
        if activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif activation == 'tanh':
            return np.tanh(x)
        elif activation == 'relu':
            return np.maximum(0, x)
        elif activation == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)
        elif activation == 'softmax':
            exps = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exps / np.sum(exps, axis=1, keepdims=True)
        return x
    
    def _activate_derivative(self, x, activation):
        """Derivative of activation function"""
        if activation == 'sigmoid':
            s = self._activate(x, 'sigmoid')
            return s * (1 - s)
        elif activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif activation == 'relu':
            return np.where(x > 0, 1, 0)
        elif activation == 'leaky_relu':
            return np.where(x > 0, 1, 0.01)
        return np.ones_like(x)
    
    def forward(self, X):
        """Forward pass through the network"""
        self.layer_inputs = []
        self.layer_outputs = [X]
        
        for i in range(len(self.weights)):
            # Input to current layer
            layer_input = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            self.layer_inputs.append(layer_input)
            
            # Output from current layer
            if i == len(self.weights) - 1:
                # Output layer
                layer_output = self._activate(layer_input, self.output_activation)
            else:
                # Hidden layer
                layer_output = self._activate(layer_input, self.activation)
            
            self.layer_outputs.append(layer_output)
        
        return self.layer_outputs[-1]
    
    def backward(self, X, y):
        """Backward pass to update weights"""
        m = X.shape[0]
        
        # For binary classification
        if self.output_activation == 'sigmoid':
            output_error = self.layer_outputs[-1] - y
        else:
            # For other problems, should be customized
            output_error = self.layer_outputs[-1] - y
        
        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1:
                # Output layer
                delta = output_error
            else:
                # Hidden layers
                delta = np.dot(delta, self.weights[i+1].T) * self._activate_derivative(self.layer_inputs[i], self.activation)
            
            # Compute gradients
            dW = np.dot(self.layer_outputs[i].T, delta) / m + (self.regularization * self.weights[i])
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            # Update weights and biases
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
    
    def fit(self, X, y, epochs=1000, batch_size=32, verbose=True):
        """Train the neural network"""
        m = X.shape[0]
        history = {'loss': []}
        
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch gradient descent
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward and backward pass
                self.forward(X_batch)
                self.backward(X_batch, y_batch)
            
            # Compute loss for tracking
            y_pred = self.forward(X)
            loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
            history['loss'].append(loss)
            
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        
        return history
    
    def predict(self, X):
        """Predict class labels"""
        probabilities = self.forward(X)
        if self.output_activation == 'sigmoid':
            return (probabilities >= 0.5).astype(int)
        else:
            return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        return self.forward(X)
