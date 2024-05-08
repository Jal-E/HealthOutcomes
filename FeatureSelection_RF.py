import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, f1_score

# Load the preprocessed dataset
df = pd.read_csv('processed_dataset.csv')

# Separate the features and the target variable
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)  # Converting back to DataFrame to keep column names

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Define perfromance metrics
metrics = {
    'Accuracy': accuracy_score,
    'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
    'F1 Score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
}

# Function to train and evaluate classifiers using cross-validation
def evaluate_classifiers_cv(X, y, metrics):
    results = {}
    for name, clf in classifiers.items():
        results[name] = {}
        for metric_name, metric in metrics.items():
            score = cross_val_score(clf, X, y, scoring=make_scorer(metric), cv=5).mean()
            results[name][metric_name] = score
    return results

# Evaluate classifiers BEFORE feature selection using cross-validation
results_before = evaluate_classifiers_cv(X_scaled, y, metrics)

# Feature selection based on Random Forest importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
features = X.columns
threshold = 0.06
selected_features = [features[i] for i in range(len(importances)) if importances[i] >= threshold]
X_selected = X_scaled[selected_features]

X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(
    X_selected, y, test_size=0.2, random_state=42)

# Evaluate classifiers AFTER feature selection using cross-validation
results_after = evaluate_classifiers_cv(X_selected, y, metrics)

# Function to save errors only for the best-performing model acc. f1 score
def save_errors_best_model():
    best_model_name = max(results_after, key=lambda x: results_after[x]['F1 Score'])
    best_model = classifiers[best_model_name]
    best_model.fit(X_train_selected, y_train_selected)
    predictions = best_model.predict(X_test_selected)
    misclassified_indices = np.where(predictions != y_test_selected)[0]
    misclassified_data = X_test_selected.iloc[misclassified_indices].copy()
    misclassified_data.loc[:, 'True Label'] = y_test_selected.iloc[misclassified_indices].values
    misclassified_data.loc[:, 'Predicted Label'] = predictions[misclassified_indices]
    misclassified_data.loc[:, 'Model'] = best_model_name
    misclassified_data.to_csv(f'../HealthOutcome/{best_model_name}_errors_after_feature_selection.csv', index=False)
    print(f"Error details for {best_model_name} saved to CSV.")

save_errors_best_model()

# Visualization and print of results
def plot_results(results_before, results_after, metric):
    labels = list(results_before.keys())
    before_values = [results_before[label][metric] for label in labels]
    after_values = [results_after[label][metric] for label in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, before_values, width, label='Before')
    rects2 = ax.bar(x + width / 2, after_values, width, label='After')

    ax.set_ylabel(metric)
    ax.set_title(f'{metric} by Model and Feature Selection')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.show()

    for label in labels:
        print(f"{label} {metric} Before: {results_before[label][metric]:.4f}, After: {results_after[label][metric]:.4f}")

# Plot for selected metrics
for metric in ['Accuracy', 'Precision', 'F1 Score']:
    plot_results(results_before, results_after, metric)
