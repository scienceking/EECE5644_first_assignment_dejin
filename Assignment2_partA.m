%% This code is for the first assignment of EECE 5644, the second question: part A.
% Clear the workspace
clear;
clc;

% Set the random seed for reproducibility
rng(12345);

% Gaussian distribution parameters for Class 1
mu1 = [0, 0, 0];          % Mean vector
sigma1 = diag([1, 1, 1]);  % Diagonal covariance matrix

% Gaussian distribution parameters for Class 2
mu2 = [2, 2, 2];          % Mean vector
sigma2 = diag([1, 1, 1]); % Diagonal covariance matrix

% Two Gaussian distribution parameters for Class 3 (mixture)
mu3_1 = [-3, -3, -3];     % Mean vector of the first Gaussian
sigma3_1 = diag([1, 1, 1]); % Diagonal covariance matrix

mu3_2 = [3, -3, 3];       % Mean vector of the second Gaussian
sigma3_2 = diag([1, 1, 1]); % Diagonal covariance matrix

% Class priors
prior1 = 0.3;
prior2 = 0.3;
prior3 = 0.4;

% Total number of samples
total_samples = 10000;

% Generate samples for Class 1
num_samples_class1 = round(total_samples * prior1);
samples_class1 = mvnrnd(mu1, sigma1, num_samples_class1);

% Generate samples for Class 2
num_samples_class2 = round(total_samples * prior2);
samples_class2 = mvnrnd(mu2, sigma2, num_samples_class2);

% Generate samples for Class 3 (from a mixture of two Gaussians)
num_samples_class3 = round(total_samples * prior3);
samples_class3_1 = mvnrnd(mu3_1, sigma3_1, floor(num_samples_class3 / 2));
samples_class3_2 = mvnrnd(mu3_2, sigma3_2, ceil(num_samples_class3 / 2));
samples_class3 = [samples_class3_1; samples_class3_2];

% Combine all samples
samples = [samples_class1; samples_class2; samples_class3];

% Class labels
labels = [ones(num_samples_class1, 1); 2 * ones(num_samples_class2, 1); 3 * ones(num_samples_class3, 1)];

% Classify all samples
predictions = zeros(size(samples, 1), 1);
for i = 1:size(samples, 1)
    predictions(i) = classify(samples(i, :), mu1, sigma1, mu2, sigma2, ...
                              mu3_1, sigma3_1, mu3_2, sigma3_2, prior1, prior2, prior3);
end

% Compute confusion matrix
confusion_matrix = confusionmat(labels, predictions);

% Display confusion matrix
disp('Confusion Matrix:');
disp(confusion_matrix);

% Plot 3D scatter plot, marking correctly and incorrectly classified data points
figure;
hold on;

% Class 1: dot for correct, red dot for incorrect
scatter3(samples(predictions == 1 & labels == 1, 1), samples(predictions == 1 & labels == 1, 2), samples(predictions == 1 & labels == 1, 3), 'go', 'DisplayName', 'Class 1 Correct');
scatter3(samples(predictions ~= 1 & labels == 1, 1), samples(predictions ~= 1 & labels == 1, 2), samples(predictions ~= 1 & labels == 1, 3), 'ro', 'DisplayName', 'Class 1 Incorrect');

% Class 2: circle for correct, red circle for incorrect
scatter3(samples(predictions == 2 & labels == 2, 1), samples(predictions == 2 & labels == 2, 2), samples(predictions == 2 & labels == 2, 3), 'gs', 'DisplayName', 'Class 2 Correct');
scatter3(samples(predictions ~= 2 & labels == 2, 1), samples(predictions ~= 2 & labels == 2, 2), samples(predictions ~= 2 & labels == 2, 3), 'rs', 'DisplayName', 'Class 2 Incorrect');

% Class 3: triangle for correct, red triangle for incorrect
scatter3(samples(predictions == 3 & labels == 3, 1), samples(predictions == 3 & labels == 3, 2), samples(predictions == 3 & labels == 3, 3), 'g^', 'DisplayName', 'Class 3 Correct');
scatter3(samples(predictions ~= 3 & labels == 3, 1), samples(predictions ~= 3 & labels == 3, 2), samples(predictions ~= 3 & labels == 3, 3), 'r^', 'DisplayName', 'Class 3 Incorrect');

xlabel('X1');
ylabel('X2');
zlabel('X3');
legend();
title('3D Scatter Plot of Classifications');
grid on;
view(3); % Ensure the plot is in 3D view
hold off;

% Function definitions must go here
% Define Gaussian distribution's probability density function
function p = gaussian_pdf(x, mu, sigma)
    d = length(mu);  % Dimension
    norm_const = 1 / ((2 * pi)^(d / 2) * sqrt(det(sigma)));
    x_mu = x' - mu';  % Ensure both x and mu are column vectors
    exponent = -0.5 * x_mu' * inv(sigma) * x_mu;
    p = norm_const * exp(exponent);
end

% Define classifier function
function label = classify(x, mu1, sigma1, mu2, sigma2, mu3_1, sigma3_1, mu3_2, sigma3_2, prior1, prior2, prior3)
    % Compute posterior probability for each class
    post1 = prior1 * gaussian_pdf(x, mu1, sigma1);
    post2 = prior2 * gaussian_pdf(x, mu2, sigma2);
    post3 = prior3 * (0.5 * gaussian_pdf(x, mu3_1, sigma3_1) + 0.5 * gaussian_pdf(x, mu3_2, sigma3_2));
    
    % Choose the class with the highest posterior probability as the classification result
    [~, label] = max([post1, post2, post3]);
end
