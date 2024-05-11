import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Load your preprocessed dataset
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

# Initialize classifiers including the Stacking Classifier
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=7))
]
stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)

classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'Stacking Classifier': stacking_classifier
}

# Define performance metrics
base_metrics = {
    'Accuracy': accuracy_score,
    'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
    'F1 Score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
}

# Function to train and evaluate classifiers using cross-validation
def evaluate_classifiers_cv(X, y, classifiers, metrics, k_values):
    results = {}
    for k in k_values:
        results[k] = {}
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        for name, clf in classifiers.items():
            results[k][name] = {}
            for metric_name, metric_func in metrics.items():
                score = cross_val_score(clf, X, y, cv=kf, scoring=make_scorer(metric_func)).mean()
                results[k][name][metric_name] = score
    return results

# Multiple k values for robust cross-validation
k_values = [10, 20, 30]

# Evaluate classifiers across multiple k values
results_k_variation = evaluate_classifiers_cv(X_scaled, y, classifiers, base_metrics, k_values)

# Print initial model performances
print("\nInitial Model Performance:")
for k in k_values:
    print(f"\nPerformance with k={k} folds:")
    for classifier, metrics in results_k_variation[k].items():
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

# Brute-force feature selection to find the best feature subset
def find_best_feature_order(X, y, classifiers, metrics, max_features=5):
    best_feature_order = None
    best_mean_accuracy = 0
    iteration_count = 0
    total_iterations = sum(1 for _ in range(1, max_features + 1) for _ in combinations(X.columns, _))
    print(f"Total combinations to test: {total_iterations}")
    for r in range(1, max_features + 1):
        for feature_order in combinations(X.columns, r):
            iteration_count += 1
            model = DecisionTreeClassifier(random_state=42)
            scores = cross_val_score(model, X[list(feature_order)], y, cv=5)
            mean_accuracy = scores.mean()
            if mean_accuracy > best_mean_accuracy:
                best_mean_accuracy = mean_accuracy
                best_feature_order = feature_order
                print(f"New best feature subset: {best_feature_order} with accuracy: {best_mean_accuracy:.4f}")
    return list(best_feature_order) if best_feature_order else list(X.columns)

best_feature_order = find_best_feature_order(X_selected, y, classifiers, base_metrics, len(selected_features))
print("\nBest Feature Order:", best_feature_order)

X_best_selected = X_selected[best_feature_order] if best_feature_order else X_selected

# Evaluate classifiers AFTER brute-force feature selection using cross-validation
results_after = evaluate_classifiers_cv(X_best_selected, y, classifiers, base_metrics, k_values)

print("\nPerformance After Feature Selection:")
for k in k_values:
    print(f"With k={k} folds:")
    for classifier, metrics in results_after[k].items():
        print(f"{classifier}: {metrics}")


# Save errors only for the best-performing model according to F1 score
def save_errors_best_model(X_test, y_test, classifiers, results):
    best_f1 = 0
    best_model_name = None
    for k in results:
        for model in results[k]:
            if 'F1 Score' in results[k][model] and results[k][model]['F1 Score'] > best_f1:
                best_f1 = results[k][model]['F1 Score']
                best_model_name = model
    if best_model_name is None:
        print("No best model found based on F1 Score.")
        return
    best_model = classifiers[best_model_name]
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)
    misclassified_indices = np.where(predictions != y_test)[0]
    misclassified_data = X_test.iloc[misclassified_indices]
    misclassified_data['True Label'] = y_test.iloc[misclassified_indices]
    misclassified_data['Predicted Label'] = predictions[misclassified_indices]
    misclassified_data['Model'] = best_model_name
    misclassified_data.to_csv(f'{best_model_name}_errors.csv', index=False)
    print(f"\nError details saved for model: {best_model_name}")

save_errors_best_model(X_test, y_test, classifiers, results_after)


# Plot results
def plot_results(results, metric):
    labels = results[10].keys()
    ks = sorted(results.keys())
    fig, ax = plt.subplots(figsize=(12, 8))
    for classifier in labels:
        metric_values = [results[k][classifier][metric] for k in ks]
        ax.plot(ks, metric_values, '-o', label=classifier)
    ax.set_xlabel('Number of folds (k)')
    ax.set_ylabel(metric)
    ax.set_title(f'Variation of {metric} with k folds')
    ax.legend()
    plt.show()

plot_results(results_after, 'Accuracy')
plot_results(results_after, 'Precision')
plot_results(results_after, 'F1 Score')

# Check performance for all models including Stacking Classifier
print("\nComparison with Paper's Baseline Results:")
paper_baselines = {'Random Forest': 97.64, 'KNN': 87.23, 'SVM': 87.47, 'Logistic Regression': 0, 'Stacking Classifier': 97.87

for model_name in classifiers.keys():  # Using classifiers dictionary to ensure all models are covered
    your_best = max((results_k_variation[k].get(model_name, {}).get('Accuracy', 0) for k in [10, 20, 30]), default=0) * 100
    print(f"{model_name:<20} | {your_best:>18.2f}% | {paper_baselines.get(model_name, 'N/A'):>15}%")

