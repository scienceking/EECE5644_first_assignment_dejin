# This code is for the first assignment of EECE 5644, the third question:
# second data: Human Activity Recognition dataset data.
import numpy as np  # Using NumPy instead of CuPy
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt

def estimate_mean_covariance(data, class_label):
    # Select data corresponding to the given class label
    class_data = data[data['activity'] == class_label]

    # Remove the activity label column, keeping the feature columns
    features = np.array(class_data.drop('activity', axis=1).values)  # Use NumPy arrays

    # Calculate sample mean vector
    mean_vector = np.mean(features, axis=0)

    # Calculate covariance matrix
    covariance_matrix = np.cov(features, rowvar=False)

    return mean_vector, covariance_matrix


def estimate_class_prior(data, class_label):
    total_samples = len(data)
    class_samples = len(data[data['activity'] == class_label])

    return class_samples / total_samples


def regularize_covariance(cov_matrix, lambd=1e-6):
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

    # Check if determinant is too close to zero to avoid division by zero
    if det < 1e-6:
        det = 1e-6  # Avoid division by zero

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


def visualize_data(features, true_labels, predicted_labels):
    """Visualize the first two features of the data with true labels and predicted labels"""
    features_np = np.array(features)  # Ensure NumPy arrays for visualization
    true_labels_np = np.array(true_labels)
    predicted_labels_np = np.array(predicted_labels)

    plt.figure(figsize=(12, 6))

    # Plot with true labels
    plt.subplot(1, 2, 1)
    plt.scatter(features_np[:, 0], features_np[:, 1], c=true_labels_np, cmap='viridis', s=10)
    plt.title("True Labels")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # Plot with predicted labels
    plt.subplot(1, 2, 2)
    plt.scatter(features_np[:, 0], features_np[:, 1], c=predicted_labels_np, cmap='viridis', s=10)
    plt.title("Predicted Labels")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.tight_layout()
    plt.show()


def save_classification_results(true_labels, predicted_labels, filename="classification_results.csv"):
    """Save true and predicted labels to a CSV file"""
    df = pd.DataFrame({
        'True Label': np.array(true_labels),  # Convert to NumPy array
        'Predicted Label': np.array(predicted_labels)  # Convert to NumPy array
    })
    df.to_csv(filename, index=False)
    print(f"Classification results saved to {filename}")


if __name__ == "__main__":
    # Specify file paths for test and train datasets
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

    # Set the alpha value for regularization
    alpha = 0.1

    # Get all unique activity labels
    unique_labels = combined_data['activity'].unique()

    # Initialize a dictionary to store the mean vector, covariance matrix, and prior probability for each class
    class_data = {}

    # Loop through all class labels
    for class_label in unique_labels:
        # Compute the mean vector and covariance matrix for this class
        mean_vector, covariance_matrix = estimate_mean_covariance(combined_data, class_label)

        # Use the formula to compute the λ value
        lambda_value = compute_lambda(covariance_matrix, alpha)

        # Regularize the covariance matrix
        regularized_covariance_matrix = regularize_covariance(covariance_matrix, lambda_value)

        # Compute the prior probability for this class
        class_prior = estimate_class_prior(combined_data, class_label)

        # Store the results in the dictionary
        class_data[class_label] = (mean_vector, regularized_covariance_matrix, class_prior)

    # Classify all samples
    true_labels = []
    predicted_labels = []

    features = np.array(combined_data.drop('activity', axis=1).values)  # Use NumPy array

    # Use tqdm to display progress bar
    for i in tqdm(range(len(features)), desc="Classifying samples"):
        sample = features[i]
        true_label = combined_data.iloc[i]['activity']
        predicted_label = classify_sample(sample, class_data)

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    # Convert true_labels and predicted_labels to NumPy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Calculate error rate
    errors = np.sum(true_labels != predicted_labels)
    error_probability = errors / len(true_labels)

    # Output error rate
    print(f"Error Probability: {error_probability * 100:.2f}%")

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Visualize the first two features
    visualize_data(features[:, :2], true_labels, predicted_labels)

    # Save classification results to CSV
    save_classification_results(true_labels, predicted_labels)
