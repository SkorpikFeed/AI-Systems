import numpy as np
import matplotlib.pyplot as plt


def visualize_classifier(classifier, X, y, title='Classifier Visualization'):
    """
    Visualize the decision boundaries of a classifier.
    
    Parameters:
    -----------
    classifier : trained classifier object
        The classifier with a predict method
    X : array-like, shape (n_samples, 2)
        Training data points
    y : array-like, shape (n_samples,)
        Target labels
    title : str, optional
        Title for the plot
    """
    # Define the minimum and maximum values for X and Y
    # with some padding
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    
    # Define the step size for the mesh
    mesh_step_size = 0.01
    
    # Define the mesh grid
    x_vals, y_vals = np.meshgrid(
        np.arange(min_x, max_x, mesh_step_size),
        np.arange(min_y, max_y, mesh_step_size)
    )
    
    # Run the classifier on the mesh
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)
    
    # Create the plot
    plt.figure()
    plt.title(title)
    
    # Specify the colors for different classes
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.Paired, shading='auto')
    
    # Overlay the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black',
                linewidth=1, cmap=plt.cm.Paired)
    
    # Specify the boundaries of the plot
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    
    # Specify the ticks on the X and Y axes
    plt.xticks(np.arange(int(min_x), int(max_x), 1.0))
    plt.yticks(np.arange(int(min_y), int(max_y), 1.0))
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()