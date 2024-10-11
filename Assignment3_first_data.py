# This code is for the first assignment of EECE 5644,
# the third question: first data：first data: wine data.
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def estimate_mean_covariance(data, class_label):
    # Select the data corresponding to the given class label
    class_data = data[data['quality'] == class_label]

    # Drop the quality label column, keeping only the feature columns
    features = class_data.drop('quality', axis=1).values

    # Calculate the sample mean vector
    mean_vector = np.mean(features, axis=0)

    # Calculate the covariance matrix
    covariance_matrix = np.cov(features, rowvar=False)

    return mean_vector, covariance_matrix


def estimate_class_prior(data, class_label):
    total_samples = len(data)
    class_samples = len(data[data['quality'] == class_label])

    return class_samples / total_samples


def regularize_covariance(cov_matrix, lambd):
    # Add regularization term λI
    identity_matrix = np.identity(cov_matrix.shape[0])
    regularized_cov_matrix = cov_matrix + lambd * identity_matrix

    return regularized_cov_matrix


def compute_lambda(cov_matrix, alpha):
    # Calculate the trace of the covariance matrix (sum of diagonal elements)
    trace_value = np.trace(cov_matrix)
    rank_value = np.linalg.matrix_rank(cov_matrix)
    lambd = alpha * (trace_value / rank_value)
    return lambd


def gaussian_pdf(x, mean_vector, cov_matrix):
    """Calculate the probability density function of a Gaussian distribution"""
    size = len(x)
    det = np.linalg.det(cov_matrix)
    norm_const = 1.0 / (np.power((2 * np.pi), float(size) / 2) * np.power(det, 1.0 / 2))
    x_mu = x - mean_vector
    inv = np.linalg.inv(cov_matrix)
    result = np.dot(np.dot(x_mu.T, inv), x_mu)
    return norm_const * np.exp(-0.5 * result)


def classify_sample(sample, class_data):
    """Classify a sample based on the minimum misclassification probability criterion"""
    max_posterior = -np.inf
    best_class = None
    for class_label, (mean_vector, cov_matrix, prior) in class_data.items():
        likelihood = gaussian_pdf(sample, mean_vector, cov_matrix)
        posterior = likelihood * prior  # Posterior probability = likelihood * prior
        if posterior > max_posterior:
            max_posterior = posterior
            best_class = class_label
    return best_class


def save_classification_results(true_labels, predicted_labels, filename="classification_results_1.csv"):
    """Save true and predicted labels to a CSV file"""
    df = pd.DataFrame({
        'True Label': true_labels,
        'Predicted Label': predicted_labels
    })
    df.to_csv(filename, index=False)
    print(f"Classification results saved to {filename}")


if __name__ == "__main__":
    # Read the dataset
    file_path = r"wine+quality\winequality-white.csv"
    data = pd.read_csv(file_path, delimiter=';')


    # Set alpha value for regularization
    alpha = 0.1

    # Get all unique quality labels
    unique_labels = data['quality'].unique()
    print(unique_labels )

    # Initialize a dictionary to store the mean vector, covariance matrix, and prior probability for each class
    class_data = {}

    # Loop through all class labels
    for class_label in unique_labels:
        # Compute the mean vector and covariance matrix for this class
        mean_vector, covariance_matrix = estimate_mean_covariance(data, class_label)

        # Use the formula to compute the λ value
        lambda_value = compute_lambda(covariance_matrix, alpha)

        # Regularize the covariance matrix
        regularized_covariance_matrix = regularize_covariance(covariance_matrix, lambda_value)

        # Compute the prior probability for this class
        class_prior = estimate_class_prior(data, class_label)

        # Store the results in the dictionary
        class_data[class_label] = (mean_vector, regularized_covariance_matrix, class_prior)

    # Classify all samples
    true_labels = []
    predicted_labels = []

    features = data.drop('quality', axis=1).values
    for i in range(len(features)):
        sample = features[i]
        true_label = data.iloc[i]['quality']
        predicted_label = classify_sample(sample, class_data)

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    # Calculate the number of misclassified samples
    errors = sum(np.array(true_labels) != np.array(predicted_labels))
    error_probability = errors / len(true_labels)

    # Output the error rate
    print(f"Error Probability: {error_probability * 100:.2f}%")

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Save the classification results to a CSV file
    save_classification_results(true_labels, predicted_labels)
