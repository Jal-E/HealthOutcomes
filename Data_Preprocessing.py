import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
df = pd.read_csv('../HealthOutcome/ObesityDataSet_raw_and_data_sinthetic.csv')

# Mapping binary variables
binary_map = {'yes': 1, 'no': 0}
df['family_history_with_overweight'] = df['family_history_with_overweight'].map(binary_map)
df['FAVC'] = df['FAVC'].map(binary_map)
df['SMOKE'] = df['SMOKE'].map(binary_map)
df['SCC'] = df['SCC'].map(binary_map)

# Manual ordinal encoding
caec_order = {'No': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
calc_order = {'No': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
df['CAEC'] = df['CAEC'].map(caec_order)
df['CALC'] = df['CALC'].map(calc_order)

# One-hot encoding for nominal variables
df = pd.get_dummies(df, columns=['Gender', 'MTRANS'])

# Label encoding the target variable
label_encoder = LabelEncoder()
df['NObeyesdad'] = label_encoder.fit_transform(df['NObeyesdad'])

# Standardizing numeric variables
scaler = StandardScaler()
numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Check for missing values
if df.isnull().sum().any():
    df = df.fillna(df.mean())  # Replace NaNs with mean of the column

# Save the processed DataFrame
df.to_csv('processed_dataset.csv', index=False)

print("Processed data has been saved to 'processed_dataset.csv'.")



'''import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import category_encoders as ce  # Import the category_encoders library

# Load your dataset
df = pd.read_csv('../HealthOutcome/ObesityDataSet_raw_and_data_sinthetic.csv')

# Mapping binary categorical variables
binary_map = {'yes': 1, 'no': 0}
df['family_history_with_overweight'] = df['family_history_with_overweight'].map(binary_map)
df['FAVC'] = df['FAVC'].map(binary_map)
df['SMOKE'] = df['SMOKE'].map(binary_map)
df['SCC'] = df['SCC'].map(binary_map)

# Binary encoding for multi-class categorical variables
categorical_cols = ['Gender', 'CAEC', 'CALC', 'MTRANS']  # Specify categorical columns
encoder = ce.BinaryEncoder(cols=categorical_cols)  # Initialize the BinaryEncoder
df = encoder.fit_transform(df)  # Apply the encoder

# Label encoding the target variable
label_encoder = LabelEncoder()
df['NObeyesdad'] = label_encoder.fit_transform(df['NObeyesdad'])

# Dictionary to keep track of label encoding for NObeyesdad
obesity_levels = {
    index: label for index, label in enumerate(label_encoder.classes_)
}

# Standardizing numeric variables
scaler = StandardScaler()
numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']  # Numeric columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Check for any missing values and fill or drop
if df.isnull().sum().any():
    print("Missing values found in the dataset.")
    df = df.fillna(df.mean())  # Replace NaNs with mean of the column, adjust strategy as needed

# Save the processed DataFrame
df.to_csv('processed_dataset.csv', index=False)  # Save the DataFrame to a CSV file

# Print mappings and confirm the action
print("Obesity levels encoding map:", obesity_levels)
print("Processed data has been saved to 'processed_dataset.csv'.")


#ONE HOT
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load your dataset
df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# Mapping binary categorical variables
binary_map = {'yes': 1, 'no': 0}
df['family_history_with_overweight'] = df['family_history_with_overweight'].map(binary_map)
df['FAVC'] = df['FAVC'].map(binary_map)
df['SMOKE'] = df['SMOKE'].map(binary_map)
df['SCC'] = df['SCC'].map(binary_map)

# One-hot encoding multi-class categorical variables
categorical_cols = ['Gender', 'CAEC', 'CALC', 'MTRANS']  # Add any other categorical columns
df = pd.get_dummies(df, columns=categorical_cols)

# Label encoding the target variable
label_encoder = LabelEncoder()
df['NObeyesdad'] = label_encoder.fit_transform(df['NObeyesdad'])

# Dictionary to keep track of label encoding for NObeyesdad
obesity_levels = {
    index: label for index, label in enumerate(label_encoder.classes_)
}

# Standardizing numeric variables
scaler = StandardScaler()
numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']  # Numeric columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Check for any missing values and fill or drop
if df.isnull().sum().any():
    print("Missing values found in the dataset.")
    df = df.fillna(df.mean())  # Replace NaNs with mean of the column, adjust strategy as needed

# Save the processed DataFrame
df.to_csv('processed_dataset.csv', index=False)  # Save the DataFrame to a CSV file

# Print mappings and confirm the action
print("Obesity levels encoding map:", obesity_levels)
print("Processed data has been saved to 'processed_dataset.csv'.")
'''