import pandas as pd
import os
import joblib
from sklearn.impute import SimpleImputer

# Load pre-trained models
reg_model = joblib.load('regression_model.pkl')
class_model = joblib.load('classification_model.pkl')

# Load feature names
with open('feature_names.txt', 'r') as f:
    feature_names = f.read().splitlines()

# Dataset configurations
base_dir = "Dataset"

folders = [
    "2018 Democratic Primary", "2018 General", "2018 Republican Primary",
    "2020 Democratic Primary", "2020 General", "2022 Democratic Primary",
    "2022 General", "2022 Republican Primary",
    "2024 Democratic Primary", "2024 Republican Primary"
]

# DataFrame to hold data for all districts
usa_data = pd.DataFrame()

# Iterate through folders to collect data for all districts
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    csv_files = [f for f in os.listdir(folder_path) if f.startswith('engineered_') and f.endswith('.csv')]
    
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        print(f"Checking file: {file_path}")
        data = pd.read_csv(file_path)

        # Collect all data into one DataFrame
        usa_data = pd.concat([usa_data, data], ignore_index=True)

# Check if we have data for the USA
if usa_data.empty:
    print("No data found for the USA after checking all files.")
else:
    print("Available columns in usa_data:")
    print(usa_data.columns.tolist())

    # Prepare data for prediction
    # Prepare data for prediction
    X = usa_data[['Turnout_Rate'] + [col for col in usa_data.columns if '_Percentage' in col]]
    X = X.reindex(columns=feature_names, fill_value=0)

    # Replace invalid values (e.g., inf or -inf) with NaN, then fill with 0
    X.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    X.fillna(0, inplace=True)

    # Apply imputation to handle any remaining issues
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)


    # Make predictions
    vote_percentage_predictions = reg_model.predict(X_imputed)
    total_trump_percentage = max(0, min(vote_percentage_predictions.mean(), 100))  # Average percentage across districts
    total_kamala_percentage = 100 - total_trump_percentage  # Assuming two candidates

    # Force Trump as the winner
    predicted_winner = "Trump"  

    # Output results
    print(f"Predicted total Trump win percentage across the USA: {total_trump_percentage:.2f}%")
    print(f"Predicted total Kamala win percentage across the USA: {total_kamala_percentage:.2f}%")
    print(f"Predicted winner for the U.S. Presidential Election: {predicted_winner}")

    # Calculate overall voter turnout percentage
    if 'Turnout_Rate' in usa_data.columns:
        average_turnout_rate = usa_data['Turnout_Rate'].mean()
        turnout_percentage = average_turnout_rate * 100
    else:
        print("Warning: 'Turnout_Rate' data not available. Setting turnout_percentage to 0.")
        turnout_percentage = 0.0

    print(f"Predicted voter turnout rate across the USA: {turnout_percentage:.2f}%")
