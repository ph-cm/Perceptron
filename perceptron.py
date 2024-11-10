import numpy as np
import random
import matplotlib.pyplot as plt

def train(positive_examples, negative_examples, num_iterations = 1000, eta = 0.01, plot_interval=200):
    # Initialize weights (almost randomly :))
    weights = np.array([0.0, 0.0, 0.0])
    
    # Plot positive and negative examples
    pos_x = [x[1] for x in positive_examples]
    pos_y = [x[2] for x in positive_examples]
    neg_x = [x[1] for x in negative_examples]
    neg_y = [x[2] for x in negative_examples]
    plt.scatter(pos_x, pos_y, color='blue', label="Positive Examples")
    plt.scatter(neg_x, neg_y, color='red', label="Negative Examples")
    
    for i in range(num_iterations):
        pos = random.choice(positive_examples)
        neg = random.choice(negative_examples)
        
        # Compute perceptron output for positive example
        z = np.dot(pos, weights)
        # Positive example classified as negative
        if z < 0:
            weights = weights + eta * np.array(pos)
        
        # Compute perceptron output for negative example
        z = np.dot(neg, weights)
        # Negative example classified as positive
        if z >= 0: 
            weights = weights - eta * np.array(neg)
        
        # Plot the decision boundary periodically
        if i % plot_interval == 0 or i == num_iterations - 1:
            x_vals = np.linspace(0, 1, 100)
            y_vals = -(weights[0] + weights[1] * x_vals) / weights[2]
            plt.plot(x_vals, y_vals, label=f"Iteration {i}", alpha=0.3)
    
    # Visualizing the final decision boundary
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.title("Visualizing a Perceptron with a Decision Boundary")
    plt.show()

    return weights
    
# Generate data examples
positive_examples = np.array([[1, x, y] for x, y in zip(np.random.rand(10), np.random.rand(10) + 1)])
negative_examples = np.array([[1, x, y] for x, y in zip(np.random.rand(10), np.random.rand(10) - 1)])

# Train the perceptron
weights = train(positive_examples, negative_examples, num_iterations=100, eta=0.1, plot_interval=20)
