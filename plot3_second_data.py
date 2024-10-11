# This code is for the first assignment of EECE 5644, the third question:
# second data: Human Activity Recognition dataset data.(the plot code)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# Define different markers for each class (only need six markers for six classes)
markers = ['o', 's', 'D', '^', 'v', 'p']  # Markers for six classes

x_test_path = r"human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\X_test.txt"
y_test_path = r"human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\y_test.txt"

x_train_path = r"human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\train\X_train.txt"
y_train_path = r"human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\train\y_train.txt"

# Read training data (X_train) and labels (y_train)
x_train = pd.read_csv(x_train_path, delim_whitespace=True, header=None)
y_train = pd.read_csv(y_train_path, delim_whitespace=True, header=None, names=['activity'])

# Read test data (X_test) and labels (y_test)
x_test = pd.read_csv(x_test_path, delim_whitespace=True, header=None)
y_test = pd.read_csv(y_test_path, delim_whitespace=True, header=None, names=['activity'])

# Combine X_train and y_train into a single DataFrame
train_data = pd.concat([x_train, y_train], axis=1)

# Combine X_test and y_test into a single DataFrame
test_data = pd.concat([x_test, y_test], axis=1)

# Merge the training and test data into a single dataset
combined_data = pd.concat([train_data, test_data], axis=0)

# Define a function to plot classification results
def plot_classification_results(features, true_labels, predicted_labels):
    unique_classes = np.unique(true_labels)  # Get all unique class labels
    print(f"Unique classes found: {len(unique_classes)}")  # Output the number of classes, should be six

    # Initialize the plot
    plt.figure(figsize=(10, 8))

    # Iterate over all class labels
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
    # Read the previously saved classification results
    filename = "classification_results_2.csv"  # Ensure the file path is correct
    classification_results = pd.read_csv(filename)

    # Get the true labels and predicted labels
    true_labels = classification_results['True Label'].values
    predicted_labels = classification_results['Predicted Label'].values

    # Extract the first two features
    features = combined_data.drop('activity', axis=1).values[:, :2]  # Use only the first two features

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)

    # Compute and print the error rate
    total_samples = len(true_labels)
    correct_predictions = np.trace(conf_matrix)  # The sum of diagonal elements represents the number of correct classifications
    error_rate = 1 - (correct_predictions / total_samples)

    print(f"Error Rate: {error_rate * 100:.2f}%")

    # Plot classification results
    plot_classification_results(features, true_labels, predicted_labels)
