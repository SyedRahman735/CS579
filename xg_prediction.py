import pandas as pd
import os
import joblib
from sklearn.impute import SimpleImputer

# Load pre-trained models
reg_model = joblib.load('regression_model_xgb.pkl')  # Correct filename for regression model
class_model = joblib.load('classification_model_xgb.pkl')  # Correct filename for classification model

# Load feature names
with open('feature_names.txt', 'r') as f:
    feature_names = f.read().splitlines()

# Dataset and district configuration
base_dir = "Dataset"
district_identifier = "850025"  # Texas - 32

# Folders to process
folders = [
    "2018 Democratic Primary", "2018 General", "2018 Republican Primary",
    "2020 Democratic Primary", "2020 General", "2022 Democratic Primary",
    "2022 General", "2022 Republican Primary",
    "2024 Democratic Primary", "2024 Republican Primary"
]

# Initialize dataframes for storing results
district_data = pd.DataFrame()
turnout_data = pd.DataFrame()

# Process each folder and extract district-specific data
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    csv_files = [f for f in os.listdir(folder_path) if f.startswith('engineered_') and f.endswith('.csv')]
    
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        data = pd.read_csv(file_path)

        # Ensure district identifier is consistent
        data['CNTYVTD'] = data['CNTYVTD'].astype(str)
        district_identifier = str(district_identifier)

        # Filter data for the specified district
        district_specific_data = data[data['CNTYVTD'] == district_identifier]
        
        if not district_specific_data.empty:
            district_data = pd.concat([district_data, district_specific_data], ignore_index=True)
            if 'voter data' in csv_file.lower():
                turnout_data = pd.concat([turnout_data, district_specific_data], ignore_index=True)

# If no data was found, exit
if district_data.empty:
    exit()

# Prepare feature matrix for prediction
X = district_data[['Turnout_Rate'] + [col for col in district_data.columns if '_Percentage' in col]]
X = X.reindex(columns=feature_names, fill_value=0)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Make predictions using the loaded models
vote_percentage_prediction = reg_model.predict(X_imputed)
winner_prediction = class_model.predict(X_imputed)

# Binary mapping logic: Only "Republican" or "Democrats"
if winner_prediction[0] == 0:
    winner_label = "Republican"
else:
    winner_label = "Democrats"

# Calculate voter turnout percentage
if not turnout_data.empty and 'Turnout_Rate' in turnout_data.columns:
    average_turnout_rate = turnout_data['Turnout_Rate'].mean()
    turnout_percentage = average_turnout_rate * 100
else:
    turnout_percentage = 0.0

# Output predictions
print(f"Predicted vote percentage for Republican: {vote_percentage_prediction[0]}")
print(f"Predicted winner for President in Texas - 32: {winner_label}")
print(f"Predicted voter turnout rate for Texas - 32: {turnout_percentage:.2f}%")
print(f"Predicted vote percentage for Congressional Representative: {vote_percentage_prediction[0]}")
print(f"Predicted winner for Congressional Representative in Texas - 32: {winner_label}")
