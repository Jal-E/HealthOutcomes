import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the misclassification errors from the CSV file
errors_path = 'Random Forest_errors_after_brute_force_feature_selection.csv'
df_errors = pd.read_csv(errors_path)

# Load the original dataset to get the actual scale of features and possibly the label mapping
original_data_path = 'ObesityDataSet_raw_and_data_sinthetic.csv'
df_original = pd.read_csv(original_data_path)

# Initialize StandardScaler and LabelEncoder by retraining on the original data
scaler = StandardScaler()
label_encoder = LabelEncoder()

# Fit scaler on 'Height' and 'Weight'
scaler.fit(df_original[['Height', 'Weight']])

# Inverse transform the standardized 'Height' and 'Weight'
df_errors[['Height_original', 'Weight_original']] = scaler.inverse_transform(df_errors[['Height', 'Weight']])

# Fit label encoder to original target labels to find the inverse
label_encoder.fit(df_original['NObeyesdad'])

# Inverse transform the encoded labels
df_errors['True Label_original'] = label_encoder.inverse_transform(df_errors['True Label'])
df_errors['Predicted Label_original'] = label_encoder.inverse_transform(df_errors['Predicted Label'])

# Add a 'Misclassification' column to indicate correct or incorrect predictions
df_errors['Misclassification'] = df_errors['True Label_original'] != df_errors['Predicted Label_original']

# Calculate the confusion matrix
cm = confusion_matrix(df_errors['True Label_original'], df_errors['Predicted Label_original'], labels=label_encoder.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)



# Plot using Plotly for interactive visualization
fig = px.scatter(df_errors, x='Height_original', y='Weight_original',
                 color='Misclassification',
                 labels={'Misclassification': 'Prediction Accuracy'},
                 title="Scatter Plot of Predictions",
                 hover_data=['True Label_original', 'Predicted Label_original'])
fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
fig.show()

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix of Misclassifications')
plt.xticks(rotation=90)  # Rotate labels to avoid overlap
plt.tight_layout()
plt.show()