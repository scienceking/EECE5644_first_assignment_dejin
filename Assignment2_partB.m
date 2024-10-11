%% This code is for the first assignment of EECE 5644, the second question: part B.
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

% Define the loss matrix 10
Lambda10 = [0 10 10; 1 0 10; 1 1 0];

% Define the loss matrix 100
Lambda100 = [0 100 100; 1 0 100; 1 1 0];


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

% Compute posterior probabilities for each class
posterior1 = prior1 * mvnpdf(samples, mu1, sigma1); % Class 1 posterior
posterior2 = prior2 * mvnpdf(samples, mu2, sigma2); % Class 2 posterior
posterior3 = prior3 * (0.5 * mvnpdf(samples, mu3_1, sigma3_1) + 0.5 * mvnpdf(samples, mu3_2, sigma3_2)); % Class 3 posterior

% Normalize posteriors so that they sum to 1
posterior_sum = posterior1 + posterior2 + posterior3;
posterior1 = posterior1 ./ posterior_sum;
posterior2 = posterior2 ./ posterior_sum;
posterior3 = posterior3 ./ posterior_sum;

% Combine posterior probabilities into a matrix
posterior_matrix = [posterior1, posterior2, posterior3];

% Apply ERM rule using the loss matrix % Expected risk for each class
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%expected_risk = posterior_matrix * Lambda10';
expected_risk = posterior_matrix * Lambda100';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Classify by choosing the class with the minimum expected risk
[~, predictions] = min(expected_risk, [], 2);

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

% Define Gaussian distribution's probability density function
function p = gaussian_pdf(x, mu, sigma)
    d = length(mu);  % Dimension
    norm_const = 1 / ((2 * pi)^(d / 2) * sqrt(det(sigma)));
    x_mu = x' - mu';  % Ensure both x and mu are column vectors
    exponent = -0.5 * x_mu' * inv(sigma) * x_mu;
    p = norm_const * exp(exponent);
end

