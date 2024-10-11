# This code is for the first assignment of EECE 5644,
# the third question: first data：first data: wine data.(the plot code)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Define different markers for each class (7 markers for 7 classes)
markers = ['o', 's', 'D', '^', 'v', 'p', '*']  # Markers for seven classes

# Define a function to plot classification results
def plot_classification_results(features, true_labels, predicted_labels):
    unique_classes = np.unique(true_labels)  # Get all unique class labels
    print(f"Unique classes found: {len(unique_classes)}")  # Ensure there are seven classes

    # Initialize the plot
    plt.figure(figsize=(10, 8))

    # Iterate through all class labels
    for idx, class_label in enumerate(unique_classes):
        # Find the indices that belong to the current class
        class_idx = np.where(true_labels == class_label)[0]

        # Separate the correctly classified and misclassified points
        correct_idx = class_idx[true_labels[class_idx] == predicted_labels[class_idx]]
        incorrect_idx = class_idx[true_labels[class_idx] != predicted_labels[class_idx]]

        # Plot correctly classified points in green
        plt.scatter(features[correct_idx, 0], features[correct_idx, 1],
                    color='green', marker=markers[idx % len(markers)], label=f"Class {class_label} - Correct",
                    alpha=0.7)

        # Plot misclassified points in red
        plt.scatter(features[incorrect_idx, 0], features[incorrect_idx, 1],
                    color='red', marker=markers[idx % len(markers)], label=f"Class {class_label} - Incorrect",
                    alpha=0.7)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Classification Results: Correct (Green) vs Incorrect (Red)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Read saved classification results
    filename = "classification_results_1.csv"  # Ensure the file path is correct
    classification_results = pd.read_csv(filename)

    # Get true labels and predicted labels
    true_labels = classification_results['True Label'].values
    predicted_labels = classification_results['Predicted Label'].values

    # Read the original dataset
    file_path = r"C:\Users\Lenovo\Desktop\EECE_Assignment1\wine+quality\winequality-white.csv"
    data = pd.read_csv(file_path, delimiter=';')

    # Extract the first two features
    features = data.drop('quality', axis=1).values[:, :2]  # Use only the first two features

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate and print the error rate
    total_samples = len(true_labels)
    correct_predictions = np.trace(conf_matrix)  # The sum of diagonal elements represents the number of correct classifications
    error_rate = 1 - (correct_predictions / total_samples)

    print(f"Error Rate: {error_rate * 100:.2f}%")

    # Plot classification results
    plot_classification_results(features, true_labels, predicted_labels)
