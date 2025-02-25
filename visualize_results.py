import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_boundary(model, X, y, title='Decision Boundary', ax=None, cmap='PuOr'):
    """
    Plot the decision boundary for a 2D dataset
    
    Parameters:
    - model: trained model with predict_proba method
    - X: input features (2D)
    - y: target labels
    - title: plot title
    - ax: matplotlib axis
    - cmap: colormap for decision regions
    """
    h = 0.02  # Step size in the mesh
    
    # Create color maps
    custom_cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Get model predictions
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    
    # For binary classification
    if Z.shape[1] == 1:
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.7)
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
    else:
        Z = np.argmax(Z, axis=1).reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=custom_cmap, alpha=0.7)
    
    # Plot the training points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), 
               edgecolor='k', cmap=plt.cm.RdYlBu, s=40, alpha=0.8)
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    
    # Add legend
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.add_artist(legend1)
    
    return ax

def plot_training_history(history):
    """
    Plot training history (loss over epochs)
    
    Parameters:
    - history: dictionary containing 'loss' list from model training
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'])
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    return plt.gca()

def compare_models(models, X, y, titles):
    """
    Compare multiple models on the same dataset
    
    Parameters:
    - models: list of trained models
    - X: input features
    - y: target labels
    - titles: list of titles for each model
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for i, (model, title) in enumerate(zip(models, titles)):
        plot_decision_boundary(model, X, y, title=title, ax=axes[i])
    
    plt.tight_layout()
    return fig
