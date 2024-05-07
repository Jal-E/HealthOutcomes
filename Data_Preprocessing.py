import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
df = pd.read_csv('../HealthOutcome/ObesityDataSet_raw_and_data_sinthetic.csv')

# Print out unique values in CAEC and CALC before mapping
print("Unique values in CAEC before mapping:", df['CAEC'].unique())
print("Unique values in CALC before mapping:", df['CALC'].unique())

# Mapping binary variables
binary_map = {'yes': 1, 'no': 0}
df['family_history_with_overweight'] = df['family_history_with_overweight'].map(binary_map)
df['FAVC'] = df['FAVC'].map(binary_map)
df['SMOKE'] = df['SMOKE'].map(binary_map)
df['SCC'] = df['SCC'].map(binary_map)

# Manual ordinal encoding, updated to include 'no'
caec_order = {'No': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3, 'no': 0}
calc_order = {'No': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3, 'no': 0}
df['CAEC'] = df['CAEC'].map(caec_order)
df['CALC'] = df['CALC'].map(calc_order)

# Check for missing or unexpected values after mapping
for col in ['CAEC', 'CALC']:
    if df[col].isnull().any():
        print(f"Missing or unexpected values found in {col}.")
        print("Unexpected values in", col, ":", df[df[col].isnull()][col])

# One-hot encoding for nominal variables
df = pd.get_dummies(df, columns=['Gender', 'MTRANS'])

# Label encoding the target variable
label_encoder = LabelEncoder()
df['NObeyesdad'] = label_encoder.fit_transform(df['NObeyesdad'])

# Standardizing numeric variables
scaler = StandardScaler()
numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save the processed DataFrame
df.to_csv('processed_dataset.csv', index=False)

print("Processed data has been saved to 'processed_dataset.csv'.")
