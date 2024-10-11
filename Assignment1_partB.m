%% This code is for the first assignment of EECE 5644, the first question: part B.
clear;
clear;
clc;

% Set the random seed for reproducibility
rng(12345);

% Parameter definition
m0 = [-1; -1; -1; -1]; % Mean vector for class 0
C0 = [2 -0.5 0.3 0; -0.5 1 -0.5 0; 0.3 -0.5 1 0; 0 0 0 2]; % Covariance matrix for class 0
m1 = [1; 1; 1; 1]; % Mean vector for class 1
C1 = [1 0.3 -0.2 0; 0.3 2 0.3 0; -0.2 0.3 1 0; 0 0 0 3]; % Covariance matrix for class 1

% Modify covariance matrices to be diagonal (naive Bayes assumption)
diag_C0 = diag(diag(C0)); % Keep only the diagonal elements (variances)
diag_C1 = diag(diag(C1)); % Keep only the diagonal elements (variances)

% Class prior probabilities
P_L0 = 0.35; % P(L=0)
P_L1 = 0.65; % P(L=1)

% Generate sample data
num_samples = 10000;
X = mvnrnd(m0, C0, round(num_samples * P_L0)); % Generate samples from class 0
X = [X; mvnrnd(m1, C1, round(num_samples * P_L1))]; % Generate samples from class 1
true_labels = [zeros(round(num_samples * P_L0), 1); ones(round(num_samples * P_L1), 1)]; % True class labels

% Compute likelihood ratio using the naive Bayes assumption (diagonal covariance matrices)
likelihood_ratio_NB = mvnpdf(X, m1', diag_C1) ./ mvnpdf(X, m0', diag_C0);

% Initialize variables for ROC curve
gamma_values = logspace(-10, 20, 10000); % Gamma values from a very small value (~0) to a very large value (~infinity)
TPR_list = zeros(length(gamma_values), 1); % List to store True Positive Rates
FPR_list = zeros(length(gamma_values), 1); % List to store False Positive Rates
error_prob_list = zeros(length(gamma_values), 1); % List to store error probabilities

% Compute TPR, FPR, and error probability for each gamma value under naive Bayes assumption
for i = 1:length(gamma_values)
    gamma = gamma_values(i);
    % Classification decision based on likelihood ratio and threshold gamma
    decision = likelihood_ratio_NB > gamma;

    % Compute TP, FP, TN, FN
    TP = sum((decision == 1) & (true_labels == 1)); % True positives
    FP = sum((decision == 1) & (true_labels == 0)); % False positives
    TN = sum((decision == 0) & (true_labels == 0)); % True negatives
    FN = sum((decision == 0) & (true_labels == 1)); % False negatives

    % Compute TPR (True Positive Rate) and FPR (False Positive Rate)
    TPR = TP / (TP + FN); % Sensitivity or Recall
    FPR = FP / (FP + TN); % False Positive Rate

    % Compute error probability P(error; gamma)
    error_prob = FPR * P_L0 + (1 - TPR) * P_L1; % Weighted by prior probabilities

    % Store TPR, FPR and error probability
    TPR_list(i) = TPR;
    FPR_list(i) = FPR;
    error_prob_list(i) = error_prob;
end

% Compute the error probability for gamma = 0.35 / 0.65
gamma_fixed = P_L0 / P_L1;
decision_fixed = likelihood_ratio_NB > gamma_fixed;

% Compute TP, FP, TN, FN for gamma = 0.35 / 0.65
TP_fixed = sum((decision_fixed == 1) & (true_labels == 1)); % True positives
FP_fixed = sum((decision_fixed == 1) & (true_labels == 0)); % False positives
TN_fixed = sum((decision_fixed == 0) & (true_labels == 0)); % True negatives
FN_fixed = sum((decision_fixed == 0) & (true_labels == 1)); % False negatives

% Compute TPR, FPR, and error probability for gamma = 0.35 / 0.65
TPR_fixed = TP_fixed / (TP_fixed + FN_fixed); % True Positive Rate
FPR_fixed = FP_fixed / (FP_fixed + TN_fixed); % False Positive Rate
error_prob_fixed = FPR_fixed * P_L0 + (1 - TPR_fixed) * P_L1; % Error probability for gamma = 0.35 / 0.65

% Output the error probability and coordinates for gamma = 0.35 / 0.65
fprintf('Error Probability (Naive Bayes) for gamma = 0.35 / 0.65: %.4f\n', error_prob_fixed);
fprintf('Corresponding TPR: %.4f, FPR: %.4f\n', TPR_fixed, FPR_fixed);

% Plot ROC curve
figure;
plot(FPR_list, TPR_list, 'b-', 'LineWidth', 2);
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title('ROC Curve (Naive Bayes Assumption)');
grid on;

% Find the minimum error probability and corresponding gamma value
[min_error_prob, min_idx] = min(error_prob_list); % Find minimum value and its index
best_gamma = gamma_values(min_idx); % Corresponding gamma value

% Get the TPR and FPR for the minimum error probability
best_TPR = TPR_list(min_idx);
best_FPR = FPR_list(min_idx);

% Output the coordinates of the minimum point
fprintf('Minimum Error Probability (Naive Bayes): %.4f at Gamma: %.4f\n', min_error_prob, best_gamma);
fprintf('Corresponding TPR: %.4f, FPR: %.4f\n', best_TPR, best_FPR);

% Plot the minimum point on the ROC curve
hold on;
plot(best_FPR, best_TPR, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r'); % Use red circle to mark the point

% Adjust the position of the text to move inside the plot area
text(best_FPR + 0.02, best_TPR - 0.02, sprintf('(%.4f, %.4f)', best_FPR, best_TPR), ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', 'FontSize', 10, 'Color', 'r');

legend('ROC Curve', 'Minimum Error Point', 'Location', 'Best');
hold off;

% Plot Error Probability curve
figure;
plot(gamma_values, error_prob_list, 'r-', 'LineWidth', 2);
xlabel('Gamma');
ylabel('Error Probability P(error)');
title('Error Probability Curve (Naive Bayes)');
set(gca, 'XScale', 'log'); % Set x-axis to logarithmic scale
grid on;

% Mark the minimum error point on the error probability curve
hold on;
plot(best_gamma, min_error_prob, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');

% Add text with coordinates to the error probability curve (displaying gamma as a decimal)
text(best_gamma * 1.1, min_error_prob, sprintf('(%.4f, %.4f)', best_gamma, min_error_prob), ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 10, 'Color', 'b');

legend('Error Probability Curve', 'Minimum Error Point', 'Location', 'Best');
hold off;
