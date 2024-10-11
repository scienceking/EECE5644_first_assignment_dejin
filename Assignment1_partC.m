%% This code is for the first assignment of EECE 5644, the first question: part C.
clear;
clear;
clc;

% Set the random seed for reproducibility
rng(12345); % You can use any integer as the seed value

% Parameter definition
m0 = [-1; -1; -1; -1]; % Mean vector for class 0
C0 = [2 -0.5 0.3 0; -0.5 1 -0.5 0; 0.3 -0.5 1 0; 0 0 0 2]; % Covariance matrix for class 0
m1 = [1; 1; 1; 1]; % Mean vector for class 1
C1 = [1 0.3 -0.2 0; 0.3 2 0.3 0; -0.2 0.3 1 0; 0 0 0 3]; % Covariance matrix for class 1

% Class prior probabilities
P_L0 = 0.35; % P(L=0)
P_L1 = 0.65; % P(L=1)

% Generate sample data
num_samples = 10000;
X_class0 = mvnrnd(m0, C0, round(num_samples * P_L0)); % Generate samples from class 0
X_class1 = mvnrnd(m1, C1, round(num_samples * P_L1)); % Generate samples from class 1
X = [X_class0; X_class1]; % Combine both classes
true_labels = [zeros(size(X_class0, 1), 1); ones(size(X_class1, 1), 1)]; % True class labels

% Compute mean vectors
mu0 = mean(X_class0); % Mean vector for class 0.
mu1 = mean(X_class1); % Mean vector for class 1

% Compute within-class scatter matrix (S_W)
% Multiply by (size(X_class0, 1) - 1) to snsure an unbiased estimate of the covariance
%S_W = cov(X_class0) * (size(X_class0, 1) - 1) + cov(X_class1) * (size(X_class1, 1) - 1);
S_W = cov(X_class0) + cov(X_class1);

% Compute between-class scatter matrix (S_B)
mean_diff = (mu1 - mu0)';
S_B = (mean_diff * mean_diff');

% Compute Fisher LDA projection vector (w_LDA).
w_LDA = inv(S_W) * (mu1' - mu0');
disp(w_LDA)

% Project the samples onto the LDA direction
projected_X = X * w_LDA;

% Initialize variables for ROC curve
tau_values = linspace(min(projected_X), max(projected_X), 1000); % Threshold values
TPR_list = zeros(length(tau_values), 1); % List to store True Positive Rates
FPR_list = zeros(length(tau_values), 1); % List to store False Positive Rates
error_prob_list = zeros(length(tau_values), 1); % List to store error probabilities

% Compute TPR, FPR, and error probability for each tau value.
for i = 1:length(tau_values)
    tau = tau_values(i);
    % Classification decision based on projection and threshold tau
    decision = projected_X > tau;

    % Compute TP, FP, TN, FN
    TP = sum((decision == 1) & (true_labels == 1)); % True positives
    FP = sum((decision == 1) & (true_labels == 0)); % False positives
    TN = sum((decision == 0) & (true_labels == 0)); % True negatives
    FN = sum((decision == 0) & (true_labels == 1)); % False negatives

    % Compute TPR (True Positive Rate) and FPR (False Positive Rate)
    TPR = TP / (TP + FN); % Sensitivity or Recall
    FPR = FP / (FP + TN); % False Positive Rate
    FNR = FN / (FN + TP); % False Negative Rate (FNR)

    % Compute error probability P(error; tau)
    error_prob = FPR * P_L0 + FNR * P_L1; % Weighted by prior probabilities

    % Store TPR, FPR and error probability
    TPR_list(i) = TPR;
    FPR_list(i) = FPR;
    error_prob_list(i) = error_prob;
end

% Find the minimum error probability and corresponding tau value
[min_error_prob, min_idx] = min(error_prob_list); % Find minimum value and its index
best_tau = tau_values(min_idx); % Corresponding tau value

% Get the TPR and FPR for the minimum error probability
best_TPR = TPR_list(min_idx);
best_FPR = FPR_list(min_idx);

% Output the coordinates of the minimum point
fprintf('Minimum Error Probability (Fisher LDA): %.4f at Tau: %.4f\n', min_error_prob, best_tau);
fprintf('Corresponding TPR: %.4f, FPR: %.4f\n', best_TPR, best_FPR);

% Plot ROC curve
figure;
plot(FPR_list, TPR_list, 'b-', 'LineWidth', 2);
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title('ROC Curve (Fisher LDA)');
grid on;

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
plot(tau_values, error_prob_list, 'r-', 'LineWidth', 2);
xlabel('Tau (Threshold)');
ylabel('Error Probability P(error)');
title('Error Probability Curve (Fisher LDA)');
grid on;

% Mark the minimum error point on the error probability curve
hold on;
plot(best_tau, min_error_prob, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');

% Add text with coordinates to the error probability curve (displaying tau as a decimal)
text(best_tau * 1.1, min_error_prob, sprintf('(%.4f, %.4f)', best_tau, min_error_prob), ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 10, 'Color', 'b');

legend('Error Probability Curve', 'Minimum Error Point', 'Location', 'Best');
hold off;
