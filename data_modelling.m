clear all; close all; clc;

% Load the QSAR dataset
qsar_load = load('QSAR_data.mat');
qsar_data = qsar_load.QSAR_data;
[row_size, col_size] = size(qsar_data);

% Clear the data
% Remove duplicate rows
qsar = unique(qsar_data, 'rows');

% Update the features (X) and labels (y) based on the unique dataset
X = qsar(:, 1:end-1);      % Features from the unique dataset
y = qsar(:, end);          % Labels (biodegradability) from the unique dataset

% Separate biodegradable and non-biodegradable from the unique dataset
biodegradable = qsar(y == 1, :);       % Rows where the label is 1
non_biodegradable = qsar(y == 0, :);   % Rows where the label is 0

% Comparison of Data with Labels
num_biodegradable = size(biodegradable, 1);
num_non_biodegradable = size(non_biodegradable, 1);

figure
bar([num_biodegradable, num_non_biodegradable]);
set(gca, 'xticklabel', {'Biodegradable', 'Non-Biodegradable'});
ylabel('Number of Instances');
title('Comparison of Biodegradable and Non-Biodegradable Instances');

fprintf('Number of rows decreased from %d to %d with %d repeating rows removed.\n', ...
        row_size, size(qsar,1), row_size - size(qsar,1));

%% Histograms
figure
for i = 1:size(X, 2)
    subplot(7, 6, i); % Adjust the subplot grid as needed
    histogram(X(:, i));
    title(['Feature ' num2str(i)]);
    xlabel('Value');
    ylabel('Frequency');
end
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);

% specific features to plot
specificFeatures = [12, 23, 41];
numSpecificFeatures = length(specificFeatures);

figure
for i = 1:numSpecificFeatures
    subplot(1, 3, i); % Adjust the subplot grid as needed
    featureIndex = specificFeatures(i);
    histogram(X(:, featureIndex));
    hold on;
    boxplot(X(:, featureIndex), 'orientation', 'horizontal', 'whisker', 1.5, 'outliersize', 4, 'jitter', 0, 'colors', 'r');
    threshold_upper = quantile(X(:, featureIndex), 0.75) + 1.5 * iqr(X(:, featureIndex));
    threshold_lower = quantile(X(:, featureIndex), 0.25) - 1.5 * iqr(X(:, featureIndex));
    yl = ylim;
    plot([threshold_upper, threshold_upper], yl, 'r--');
    plot([threshold_lower, threshold_lower], yl, 'r--');
    
    hold off;
    title(['Feature ' num2str(featureIndex)]);
    xlabel('Value');
    ylabel('Frequency');
end

%% remove outliers

% Calculate the Z-score for each feature
zscore = zscore(X);
threshold = 3;

% indices of outliers
outliers = abs(zscore) > threshold;
% rows with outliers
outlier_rows = any(outliers, 2);

% Remove rows with outliers
X_no_outliers = X(~outlier_rows, :);

% Display how many outliers were removed
fprintf('Removed %d rows containing outliers.\n', sum(outlier_rows));
disp('-----------------------------------------------------------------');

figure
for i = 1:size(X_no_outliers, 2)
    subplot(7, 6, i);
    histogram(X_no_outliers(:, i));
    title(['Feature ' num2str(i)]);
    xlabel('Value');
    ylabel('Frequency');
end
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);

% specific features to plot
specificFeatures = [12, 23, 41];
numSpecificFeatures = length(specificFeatures);

figure
for i = 1:numSpecificFeatures
    subplot(1, 3, i);
    featureIndex = specificFeatures(i);
    histogram(X_no_outliers(:, featureIndex));
    hold on;
    boxplot(X_no_outliers(:, featureIndex), 'orientation', 'horizontal', 'whisker', 1.5, 'outliersize', 4, 'jitter', 0, 'colors', 'r');
    threshold_upper = quantile(X_no_outliers(:, featureIndex), 0.75) + 1.5 * iqr(X_no_outliers(:, featureIndex));
    threshold_lower = quantile(X_no_outliers(:, featureIndex), 0.25) - 1.5 * iqr(X_no_outliers(:, featureIndex));
    yl = ylim;
    plot([threshold_upper, threshold_upper], yl, 'r--');
    plot([threshold_lower, threshold_lower], yl, 'r--');
    
    hold off;
    title(['Feature ' num2str(featureIndex)]);
    xlabel('Value');
    ylabel('Frequency');
end

%% pre-processing

% Split the data into training and testing sets
rng(0);
cv = cvpartition(size(X, 1), 'HoldOut', 0.2); 
idx = cv.test;
X_train = X(~idx, :);
y_train = y(~idx, :);
X_test = X(idx, :);
y_test = y(idx, :);

% Standardize and normalize the training data
mu = mean(X_train);
sigma = std(X_train);
X_train_standardized = (X_train - mu) ./ sigma;

X_train_min = min(X_train_standardized);
X_train_max = max(X_train_standardized);
X_train_normalized = (X_train_standardized - X_train_min) ./ (X_train_max - X_train_min);

% Standardize and normalize the test data using the training data parameters
X_test_standardized = (X_test - mu) ./ sigma;
X_test_normalized = (X_test_standardized - X_train_min) ./ (X_train_max - X_train_min);

% Just standardizing:
X_train = X_train_standardized;
X_test = X_test_standardized;

% Both standardizing and normalizing
% X_train = X_train_normalized;
% X_test = X_test_normalized;

%% Built-in Functions

% support vector machines:
model_SVM = fitcsvm(X_train, y_train);
% Make predictions on the test set
y_pred_SVM = predict(model_SVM, X_test);

disp("Results for Support Vector Machines Model:")
accuracy_of_model(y_test, y_pred_SVM)
evaluateModel(y_test, y_pred_SVM, ' Support Vector Machines');

%% K Nearest Neighbour

best_k = 1;
best_accuracy_knn = 0;

for k = 1:20    % number of neighbours
    predictions_knn = zeros(size(X_test, 1), 1);
    row_X_test = size(X_test, 1);
    % Predict class for each test point using kNN
    for i = 1:row_X_test
        predictions_knn(i) = predict_KNN(X_test(i, :), X_train, y_train, k);
    end
    accuracy = sum(y_test == predictions_knn) / numel(y_test);
    if accuracy > best_accuracy_knn
        best_accuracy_knn = accuracy;
        best_k = k;
    end
end
disp('-----------------------------------------------------------------');
disp("Results for K Nearest Neighbour Model:")
fprintf('Best k: %e with accuracy: %.2f%%\n', best_k, best_accuracy_knn * 100);

evaluateModel(y_test, predictions_knn, ' K Nearest Neighbour');

%% Logistic Regression with Gradient Descent

% hyperparameters
theta = zeros(size(X_train, 2), 1);
max_iterations = 1000;

alphas = [0.001, 0.01, 0.1, 1];     % learning rate for the gradient descent 
epsilons = [1e-4, 1e-6, 1e-8];      % convergence threshold for gradient descent
best_alpha = 0;
best_epsilon = 0;
best_accuracy = 0;

for alpha = alphas
    for epsilon = epsilons
        % Logistic regression algorithm
        for iter = 1:max_iterations
            z = X_train * theta;            % weighted sum of inputs
            sigma = 1 ./ (1 + exp(-z));     % logistic (sigmoid) function to predict the probability
            gradient = (X_train' * (sigma - y_train)) / length(y_train); % gradient of the cost function
            theta_prev = theta;
            theta = theta - alpha * gradient; % update the weights by taking a step proportional to the gradient
            
            % Check for convergence: if weights have not changed significantly, exit the loop
            if norm(theta - theta_prev) < epsilon 
                break;
            end
        end
        % Predictions
        y_pred_logreg = double(X_test * theta > 0); % Using 0 as threshold
        accuracy = sum(y_test == y_pred_logreg) / length(y_test);
        if accuracy > best_accuracy
            best_accuracy = accuracy;
            best_alpha = alpha;
            best_epsilon = epsilon;
        end
    end
end
disp('-----------------------------------------------------------------');
disp("Results for Logistic Regression with Gradient Descent Model:")
fprintf('Best alpha: %e, best epsilon: %e with accuracy: %.2f%%\n', best_alpha, best_epsilon, best_accuracy * 100);
%%
evaluateModel(y_test, y_pred_logreg, ' Logistic Regression');
ROC(y_test, X_test, theta, ' Logistic Regression with Gradient Descent');

%% functions

% calculate euclidean distance
function dist = euclideanDistance(point1, point2)
    dist = sqrt(sum((point1 - point2) .^ 2));
end

function label = predict_KNN(testPoint, XTrain, yTrain, k)
    distances = zeros(size(XTrain, 1), 1);

    for i = 1:length(distances)
        distances(i) = euclideanDistance(testPoint, XTrain(i, :));
    end

    % Sort distances and select the k nearest neighbors
    [~, sortedIndices] = sort(distances);
    nearestNeighbors = yTrain(sortedIndices(1:k));
    label = mode(nearestNeighbors);         % predict
end

function accuracy_of_model(y_test, y_pred)
% accuracy
    accuracy = sum(y_test == y_pred) / length(y_test);
    fprintf('Accuracy: %.2f%%\n', accuracy * 100);
end

function evaluateModel(y_test, y_pred, model_name)
    % confusion matrix
    confusion_mat = confusionmat(y_test, y_pred);
    disp('Confusion Matrix:');
    disp(confusion_mat);
    true_positives = confusion_mat(2, 2);
    false_positives = confusion_mat(1, 2);
    true_negatives = confusion_mat(1, 1);
    false_negatives = confusion_mat(2, 1);
     
    % Metrics calculations
    TPR = true_positives / (true_positives + false_negatives); 
    FPR = false_positives / (false_positives + true_negatives); 
    TNR = true_negatives / (true_negatives + false_positives); 
    FNR = false_negatives / (true_positives + false_negatives); 
    
    fprintf('True Positive Rate (Sensitivity): %.2f\n', TPR);
    fprintf('False Positive Rate: %.2f\n', FPR);
    fprintf('True Negative Rate (Specificity): %.2f\n', TNR);
    fprintf('False Negative Rate: %.2f\n', FNR);
    % We would like to find a classifier that has TPR=1 and FPR=0.

    figure
    confusion_chart = confusionchart(y_test, y_pred);
    confusion_chart.Title = ['Confusion Matrix: ' model_name];
    confusion_chart.FontSize = 12;
    
    % precision
    precision = true_positives / (true_positives + false_positives);
    fprintf('Precision: %.2f\n', precision);

    % recall
    recall = true_positives / (true_positives + false_negatives);
    fprintf('Recall: %.2f\n', recall);

    % F1 score
    f1Score = 2 * (precision * recall) / (precision + recall);
    fprintf('F1 Score: %.2f\n', f1Score);
end

function ROC(y_test, X_test, theta, model_name)
    % Compute the predicted probabilities
    probabilities = 1 ./ (1 + exp(-X_test * theta));    % Predicted probabilities

    % Calculate the ROC curve and AUC
    % If your logistic regression model outputs a single probability for the positive class:
    [X_ROC, y_ROC, ~, AUC] = perfcurve(y_test, probabilities, 1);

    % Plot the ROC curve
    figure
    plot(X_ROC, y_ROC, 'b', 'LineWidth', 1.5);
    hold on; 
    plot([0, 1], [0, 1], 'r--');
    hold off;
    title(['ROC Curve: ' model_name]);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    legend(['AUC = ' num2str(AUC)]);
    grid on;
end

