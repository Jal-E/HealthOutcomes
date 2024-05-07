import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Load the preprocessed dataset
df = pd.read_csv('../HealthOutcome/processed_dataset.csv')

# Separate the features and the target variable
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Function to train and evaluate classifiers
def evaluate_classifiers(X_train, X_test, y_train, y_test):
    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1 Score': f1_score(y_test, y_pred, average='weighted'),
            'ROC AUC': roc_auc_score(y_test, clf.predict_proba(X_test), average='weighted', multi_class='ovr')
        }
    return results

# Evaluate classifiers before feature selection
results_before = evaluate_classifiers(X_train, X_test, y_train, y_test)

# Feature selection based on Random Forest importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
features = X.columns
threshold = 0.06
selected_features = [features[i] for i in range(len(importances)) if importances[i] >= threshold]
dropped_features = [features[i] for i in range(len(importances)) if importances[i] < threshold]
X_selected = X.loc[:, selected_features]

print("Dropped Features:", dropped_features)
print("Selected Features:", selected_features)
print("Original number of features:", len(features))
print("Number of features after selection:", len(selected_features))

# Split the data with selected features
X_train_selected, X_test_selected, _, _ = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Evaluate classifiers after feature selection
results_after = evaluate_classifiers(X_train_selected, X_test_selected, y_train, y_test)

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

    # Printing metric values before and after
    for label in labels:
        print(f"{label} {metric} Before: {results_before[label][metric]:.4f}, After: {results_after[label][metric]:.4f}")

# Plot for each metric
for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']:
    plot_results(results_before, results_after, metric)

#Only Accuracy checked
'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the preprocessed dataset
df = pd.read_csv('processed_dataset.csv')

# Separate the features and the target variable
X = df.drop('NObeyesdad', axis=1)  # assuming 'NObeyesdad' is the target
y = df['NObeyesdad']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Train and evaluate classifiers before feature selection
results_before = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results_before[name] = accuracy

# Get feature importances from Random Forest
rf = classifiers['Random Forest']
rf.fit(X_train, y_train)
importances = rf.feature_importances_
features = X.columns
feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Set a threshold for feature importance
threshold = 0.06

# Drop features with importance below the threshold
selected_features = feature_importances[feature_importances['Importance'] >= threshold]['Feature']
X_selected = X[selected_features]

# Split the data into training and test sets using selected features
X_train_selected, X_test_selected, _, _ = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train and evaluate classifiers after feature selection
results_after = {}
for name, clf in classifiers.items():
    clf.fit(X_train_selected, y_train)
    y_pred_selected = clf.predict(X_test_selected)
    accuracy_selected = accuracy_score(y_test, y_pred_selected)
    results_after[name] = accuracy_selected

# Print results
print("Accuracy before feature selection:")
for name, acc in results_before.items():
    print(f"{name}: {acc:.2f}")

print("\nAccuracy after feature selection:")
for name, acc in results_after.items():
    print(f"{name}: {acc:.2f}")'''




#ONLY RANDOM FOREST :
'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the preprocessed dataset
df = pd.read_csv('processed_dataset.csv')

# Separate the features and the target variable
X = df.drop('NObeyesdad', axis=1)  # assuming 'NObeyesdad' is the target
y = df['NObeyesdad']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the RandomForest model: {accuracy:.2f}")

# Get feature importances
importances = rf.feature_importances_
features = X.columns

# Create a DataFrame to view the features and their importance scores
feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plotting feature importances
plt.figure(figsize=(12, 8))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  # Invert the Y-axis to show the most important at the top
plt.show()

# Set a threshold for feature importance
threshold = 0.06  #best : .06

# Drop features with importance below the threshold
selected_features = feature_importances[feature_importances['Importance'] >= threshold]['Feature']
X_selected = X[selected_features]

# Split the data into training and test sets using selected features
X_train_selected, X_test_selected, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Retrain the model with selected features
rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(X_train_selected, y_train)

# Predict on the test set using selected features
y_pred_selected = rf_selected.predict(X_test_selected)

# Evaluate the model with selected features
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print(f"Accuracy of the RandomForest model with selected features: {accuracy_selected:.2f}")
'''