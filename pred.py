import pandas as pd
import os
import joblib
from sklearn.impute import SimpleImputer

# Load models
reg_model = joblib.load('regression_model.pkl')
class_model = joblib.load('classification_model.pkl')

# Load feature names
with open('feature_names.txt', 'r') as f:
    feature_names = f.read().splitlines()

base_dir = "Dataset"
district_identifier = "850025"
district_identifier = district_identifier.strip()

folders = [
    "2018 Democratic Primary", "2018 General", "2018 Republican Primary",
    "2020 Democratic Primary", "2020 General", "2022 Democratic Primary",
    "2022 General", "2022 Republican Primary",
    "2024 Democratic Primary", "2024 Republican Primary"
]

district_data = pd.DataFrame()
turnout_data = pd.DataFrame()

# Iterate through folders
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist.")
        continue

    csv_files = [f for f in os.listdir(folder_path) if f.startswith('engineered_') and f.endswith('.csv')]

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        print(f"Checking file: {file_path}")
        try:
            data = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Ensure CNTYVTD and district_identifier are comparable
        data['CNTYVTD'] = data['CNTYVTD'].astype(str).str.strip()
        district_specific_data = data[data['CNTYVTD'] == district_identifier]

        if district_specific_data.empty:
            print(f"No relevant data in {csv_file} for district {district_identifier}, skipping.")
            continue

        district_data = pd.concat([district_data, district_specific_data], ignore_index=True)
        print(f"Data for district {district_identifier} loaded successfully from {csv_file}.")

        if 'voter data' in csv_file.lower():
            turnout_data = pd.concat([turnout_data, district_specific_data], ignore_index=True)

if district_data.empty:
    print(f"No data found for district {district_identifier} after checking all files.")
else:
    print("Available columns in district_data:")
    print(district_data.columns.tolist())

    # Ensure all features from feature_names.txt are included
    missing_features = [col for col in feature_names if col not in district_data.columns]
    if missing_features:
        print(f"Adding missing features with default value 0: {missing_features}")
        for feature in missing_features:
            district_data[feature] = 0  # Add missing features with default value

    # Construct X using the exact feature list from feature_names.txt
    X = district_data[feature_names]  # Use only features defined in feature_names.txt
    imputer = SimpleImputer(strategy='mean')  # Impute missing values
    X_imputed = imputer.fit_transform(X)

    vote_percentage_prediction = reg_model.predict(X_imputed)
    vote_percentage_prediction = max(0, min(vote_percentage_prediction[0], 100))

    winner_prediction = class_model.predict(X_imputed)
    winner_mapping = {
        "TrumpR_24P_President": "Trump",
        "BidenD_24P_President": "Kamala"
    }
    predicted_winner = winner_mapping.get(winner_prediction[0], winner_prediction[0])

    print(f"Predicted Trump win percentage: {vote_percentage_prediction:.2f}%")
    print(f"Predicted Kamala win percentage: {100 - vote_percentage_prediction:.2f}%")
    print(f"Predicted winner for President in district {district_identifier}: {predicted_winner}")

    if not turnout_data.empty and 'Turnout_Rate' in turnout_data.columns:
        turnout_data['Turnout_Rate'] = pd.to_numeric(turnout_data['Turnout_Rate'], errors='coerce')
        average_turnout_rate = turnout_data['Turnout_Rate'].mean()
        turnout_percentage = average_turnout_rate * 100 if not pd.isna(average_turnout_rate) else 0
    else:
        print("Warning: 'Turnout_Rate' data not available or invalid.")
        turnout_percentage = 0.0

    print(f"Predicted voter turnout rate for district {district_identifier}: {turnout_percentage:.2f}%")
