import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the preprocessed dataset
df = pd.read_csv('../HealthOutcome/processed_dataset.csv')

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




'''import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the preprocessed data
df = pd.read_csv('processed_dataset.csv')

# Separate features and target
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (necessary for models like SVM and Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define and fit models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)
}

# Dictionary to store model scores
model_scores = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test) if hasattr(model, "predict_proba") else model.decision_function(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    auc = roc_auc_score(y_test, probabilities, multi_class='ovr', average='weighted')
    model_scores[name] = [accuracy, precision, recall, f1, auc]

# Extract and visualize feature importance for the Decision Tree
feature_importances = models['Decision Tree'].feature_importances_
features = X.columns  # Assuming X is a DataFrame

# Create a pandas Series for visualization
importances = pd.Series(data=feature_importances, index=features).sort_values(ascending=False)

# Improved plotting for feature importance
plt.figure(figsize=(12, 8))  # Larger figure size for better readability
importances.plot(kind='bar', color='teal')
plt.title('Feature Importance in Decision Tree Model')
plt.ylabel('Importance')
plt.xlabel('Features')
plt.xticks(rotation=90)  # Rotate labels to make them readable
plt.tight_layout()  # Adjust layout to make room for label rotation
plt.show()

# Convert results to DataFrame
results_df = pd.DataFrame(model_scores, index=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']).T

# Display the performance metrics
print(results_df)

# Determine the best model based on F1 Score
best_model = results_df['F1 Score'].idxmax()
print(f"Best performing model: {best_model} with F1 Score: {results_df.loc[best_model, 'F1 Score']}")

# Visualization of the results
results_df.drop('ROC AUC', axis=1).plot(kind='bar', figsize=(10, 7))
plt.title('Comparison of Model Performance')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()'''
