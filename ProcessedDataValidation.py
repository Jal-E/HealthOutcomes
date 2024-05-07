import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('../HealthOutcome/processed_dataset.csv')

# Define the ordinal and label encoding orders and mappings
caec_order = ['No', 'Sometimes', 'Frequently', 'Always']
calc_order = ['No', 'Sometimes', 'Frequently', 'Always']

# Verify Ordinal Encoding for CAEC
caec_reverse_map = {index: category for index, category in enumerate(caec_order)}
df['CAEC_decoded'] = df['CAEC'].map(caec_reverse_map)
print("Unique values in CAEC (encoded -> decoded):")
for encoded, decoded in caec_reverse_map.items():
    print(f"{encoded} -> {decoded}")
print("\nSample data for CAEC verification:")
print(df[['CAEC', 'CAEC_decoded']].head(10))

# Verify Ordinal Encoding for CALC
calc_reverse_map = {index: category for index, category in enumerate(calc_order)}
df['CALC_decoded'] = df['CALC'].map(calc_reverse_map)
print("Unique values in CALC (encoded -> decoded):")
for encoded, decoded in calc_reverse_map.items():
    print(f"{encoded} -> {decoded}")
print("\nSample data for CALC verification:")
print(df[['CALC', 'CALC_decoded']].head(10))

# Validate Label Encoding for NObeyesdad
label_encoder = LabelEncoder()
label_encoder.fit(df['NObeyesdad'])  # Simulate fitting to retrieve classes
nobeyesdad_mappings = {label: index for index, label in enumerate(label_encoder.classes_)}
print("Label encoding mappings for NObeyesdad:", nobeyesdad_mappings)

# Optional: Decode to check
df['NObeyesdad_decoded'] = label_encoder.inverse_transform(df['NObeyesdad'])
print("\nSample data for NObeyesdad verification:")
print(df[['NObeyesdad', 'NObeyesdad_decoded']].head(10))
