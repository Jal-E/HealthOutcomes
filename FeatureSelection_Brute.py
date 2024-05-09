import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, f1_score
from copy import deepcopy

# Load the preprocessed dataset
df = pd.read_csv('processed_dataset.csv')

# Separate the features and the target variable
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Define performance metrics
base_metrics = {
    'Accuracy': accuracy_score,
    'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
    'F1 Score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
}

# Function to train and evaluate classifiers using cross-validation
def evaluate_classifiers_cv(X, y, metrics):
    results = {}
    for name, clf in classifiers.items():
        results[name] = {}
        for metric_name, metric_func in metrics.items():
            if callable(metric_func):
                score = cross_val_score(clf, X, y, scoring=make_scorer(metric_func), cv=5).mean()
                results[name][metric_name] = score
            else:
                print(f"Error: Metric {metric_name} is not callable!")
    return results

# Evaluate classifiers BEFORE feature selection using cross-validation
metrics = deepcopy(base_metrics)
results_before = evaluate_classifiers_cv(X_scaled, y, metrics)
print("\nPerformance before any feature selection:")
for classifier, metrics in results_before.items():
    print(f"{classifier}: {metrics}")

# Feature selection based on Random Forest importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
features = X.columns
threshold = 0.06
selected_features = [features[i] for i in range(len(importances)) if importances[i] >= threshold]
X_selected = X_scaled[selected_features]

print("\nFeatures selected based on importance threshold:")
print(selected_features)

# Evaluate classifiers AFTER initial feature selection using cross-validation
metrics = deepcopy(base_metrics)
results_after_initial_selection = evaluate_classifiers_cv(X_selected, y, metrics)
print("\nPerformance after initial feature selection based on Random Forest importance:")
for classifier, metrics in results_after_initial_selection.items():
    print(f"{classifier}: {metrics}")

# Brute-force feature selection to find the best feature subset
def find_best_feature_order(X, y, classifiers, metrics):
    best_feature_order = None
    best_mean_accuracy = 0
    iteration_count = 0
    total_iterations = sum(1 for r in range(1, len(X.columns) + 1) for _ in combinations(X.columns, r))

    for perm_length in range(1, len(X.columns) + 1):
        for feature_order in combinations(X.columns, perm_length):
            iteration_count += 1
            model = DecisionTreeClassifier(random_state=42)
            scores = cross_val_score(model, X[list(feature_order)], y, cv=5)
            mean_accuracy = scores.mean()

            if iteration_count % 10 == 0 or mean_accuracy > best_mean_accuracy:
                print(f"Iteration: {iteration_count}/{total_iterations}, Best Accuracy: {best_mean_accuracy:.4f}")

            if mean_accuracy > best_mean_accuracy:
                best_mean_accuracy = mean_accuracy
                best_feature_order = feature_order
                print(f"New best feature subset: {best_feature_order} with accuracy: {best_mean_accuracy:.4f}")

    return best_feature_order

# Find the best feature order using brute force
best_feature_order = find_best_feature_order(X_selected, y, classifiers, base_metrics)
print("\nBest Feature Order:", best_feature_order)

# Filter X_selected with the best feature order
X_best_selected = X_selected[list(best_feature_order)]

# Split the data based on the best selected features
X_train_best, X_test_best, y_train_best, y_test_best = train_test_split(
    X_best_selected, y, test_size=0.2, random_state=42)

# Evaluate classifiers AFTER brute-force feature selection using cross-validation
metrics = deepcopy(base_metrics)
results_after = evaluate_classifiers_cv(X_best_selected, y, metrics)
print("\nPerformance after brute-force feature selection:")
for classifier, metrics in results_after.items():
    print(f"{classifier}: {metrics}")

# Save errors only for the best-performing model according to F1 score
def save_errors_best_model(X_test, y_test, classifiers, results):
    try:
        best_model_name = max(results, key=lambda x: results[x]['F1 Score'])
        best_model = classifiers[best_model_name]
        best_model.fit(X_train_best, y_train_best)
        predictions = best_model.predict(X_test)
        misclassified_indices = np.where(predictions != y_test)[0]
        misclassified_data = X_test.iloc[misclassified_indices].copy()
        misclassified_data['True Label'] = y_test.iloc[misclassified_indices].values
        misclassified_data['Predicted Label'] = predictions[misclassified_indices]
        misclassified_data['Model'] = best_model_name
        misclassified_data.to_csv(f'{best_model_name}_errors_after_brute_force_feature_selection.csv', index=False)
        print(f"\nError details for {best_model_name} saved to CSV.")
        print(f"Best Model: {best_model_name}")
        print(f"Performance: {results[best_model_name]}")
    except Exception as e:
        print(f"An error occurred: {e}")

save_errors_best_model(X_test_best, y_test_best, classifiers, results_after)

# Visualization and print of results
def plot_results(results_before, results_after_initial_selection, results_after, metric):
    labels = list(results_before.keys())
    before_values = [results_before[label][metric] for label in labels]
    after_initial_values = [results_after_initial_selection[label][metric] for label in labels]
    after_values = [results_after[label][metric] for label in labels]

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, before_values, width, label='Before Feature Selection')
    rects2 = ax.bar(x, after_initial_values, width, label='After Initial Selection')
    rects3 = ax.bar(x + width, after_values, width, label='After Brute-force Selection')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} by Model and Feature Selection Stages')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')
    ax.bar_label(rects3, padding=3, fmt='%.2f')

    fig.tight_layout()

    plt.show()

# Plot for selected metrics
for metric in ['Accuracy', 'Precision', 'F1 Score']:
    plot_results(results_before, results_after_initial_selection, results_after, metric)
