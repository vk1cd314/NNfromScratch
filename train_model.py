import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from data_generator import DataGenerator
from visualize_results import plot_decision_boundary, plot_training_history, compare_models
import os

def evaluate_model(model, X, y):
    """Evaluate model accuracy"""
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)
    return accuracy

def train_and_visualize(dataset_name='circles', hidden_layers=[16, 8], activation='tanh', 
                        learning_rate=0.01, regularization=0.01, epochs=2000):
    """
    Train neural network on a dataset and visualize results
    
    Parameters:
    - dataset_name: type of dataset ('circles', 'moons', 'spiral', 'xor')
    - hidden_layers: list of hidden layer sizes
    - activation: activation function for hidden layers
    - learning_rate: learning rate for gradient descent
    - regularization: regularization strength
    - epochs: number of training epochs
    
    Returns:
    - model: trained neural network
    - X: dataset features
    - y: dataset labels
    - history: training history
    """
    # Generate dataset
    if dataset_name == 'circles':
        X, y = DataGenerator.circles(n_samples=1000, noise=0.1)
    elif dataset_name == 'moons':
        X, y = DataGenerator.moons(n_samples=1000, noise=0.1)
    elif dataset_name == 'spiral':
        X, y = DataGenerator.spiral(n_samples=1000, noise=0.1)
    elif dataset_name == 'xor':
        X, y = DataGenerator.xor(n_samples=1000, noise=0.05)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create model architecture
    input_dim = X.shape[1]
    output_dim = 1  # Binary classification
    
    layers = [input_dim] + hidden_layers + [output_dim]
    model = NeuralNetwork(
        layers=layers,
        activation=activation,
        output_activation='sigmoid',
        learning_rate=learning_rate,
        regularization=regularization
    )
    
    # Train model
    history = model.fit(X, y, epochs=epochs, batch_size=32, verbose=True)
    
    # Evaluate model
    accuracy = evaluate_model(model, X, y)
    print(f"Accuracy on {dataset_name} dataset: {accuracy:.4f}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot decision boundary
    plot_decision_boundary(model, X, y, 
                          title=f'Decision Boundary on {dataset_name.capitalize()} Dataset\n'
                                f'Architecture: {layers}, Activation: {activation}\n'
                                f'Accuracy: {accuracy:.4f}', 
                          ax=axes[0])
    
    # Plot training history
    axes[1].plot(history['loss'])
    axes[1].set_title('Training Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)
    
    # Save the figure
    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    fig.savefig(f'results/{dataset_name}_decision_boundary.png')
    
    return model, X, y, history

def compare_activations(dataset_name='circles'):
    """Compare different activation functions on the same dataset"""
    activations = ['tanh', 'relu', 'leaky_relu']
    models = []
    
    # Generate dataset
    if dataset_name == 'circles':
        X, y = DataGenerator.circles(n_samples=1000, noise=0.1)
    elif dataset_name == 'moons':
        X, y = DataGenerator.moons(n_samples=1000, noise=0.1)
    elif dataset_name == 'spiral':
        X, y = DataGenerator.spiral(n_samples=1000, noise=0.1)
    elif dataset_name == 'xor':
        X, y = DataGenerator.xor(n_samples=1000, noise=0.05)
    
    titles = []
    
    for activation in activations:
        print(f"Training with {activation} activation...")
        model = NeuralNetwork(
            layers=[2, 16, 8, 1],
            activation=activation,
            output_activation='sigmoid',
            learning_rate=0.01,
            regularization=0.01
        )
        
        history = model.fit(X, y, epochs=1000, batch_size=32, verbose=False)
        accuracy = evaluate_model(model, X, y)
        
        models.append(model)
        titles.append(f'{activation.capitalize()}: Accuracy={accuracy:.4f}')
    
    # Compare models
    fig = compare_models(models, X, y, titles)
    
    # Save the figure
    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    fig.savefig(f'results/{dataset_name}_activation_comparison.png')

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Train on different datasets
    print("Training on Circles dataset...")
    model_circles, X_circles, y_circles, _ = train_and_visualize('circles', hidden_layers=[16, 8])
    
    print("\nTraining on Moons dataset...")
    model_moons, X_moons, y_moons, _ = train_and_visualize('moons', hidden_layers=[16, 8])
    
    print("\nTraining on Spiral dataset...")
    model_spiral, X_spiral, y_spiral, _ = train_and_visualize('spiral', hidden_layers=[32, 16])
    
    print("\nTraining on XOR dataset...")
    model_xor, X_xor, y_xor, _ = train_and_visualize('xor', hidden_layers=[8, 4])
    
    # Compare different activation functions
    print("\nComparing activation functions on Circles dataset...")
    compare_activations('circles')
    
    print("\nComparing activation functions on Spiral dataset...")
    compare_activations('spiral')
