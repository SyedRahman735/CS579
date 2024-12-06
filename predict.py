import pandas as pd
import os
import joblib
from sklearn.impute import SimpleImputer

reg_model = joblib.load('regression_model.pkl')
class_model = joblib.load('classification_model.pkl')

with open('feature_names.txt', 'r') as f:
    feature_names = f.read().splitlines()

base_dir = "Dataset"
district_identifier = "850025"  

folders = [
    "2018 Democratic Primary", "2018 General", "2018 Republican Primary",
    "2020 Democratic Primary", "2020 General", "2022 Democratic Primary",
    "2022 General", "2022 Republican Primary",
    "2024 Democratic Primary", "2024 Republican Primary"
]

district_data = pd.DataFrame()
turnout_data = pd.DataFrame()

for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    csv_files = [f for f in os.listdir(folder_path) if f.startswith('engineered_') and f.endswith('.csv')]
    
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        print(f"Checking file: {file_path}")
        data = pd.read_csv(file_path)

        data['CNTYVTD'] = data['CNTYVTD'].astype(str)
        district_identifier = str(district_identifier)

        district_specific_data = data[data['CNTYVTD'] == district_identifier]
        
        if not district_specific_data.empty:
            district_data = pd.concat([district_data, district_specific_data], ignore_index=True)
            print(f"Data for district {district_identifier} loaded successfully from {csv_file}.")
            
            if 'voter data' in csv_file.lower():
                turnout_data = pd.concat([turnout_data, district_specific_data], ignore_index=True)
        else:
            print(f"No data found in {csv_file} for district {district_identifier}")

if district_data.empty:
    print(f"No data found for district {district_identifier} after checking all files.")
else:
    print("Available columns in district_data:")
    print(district_data.columns.tolist())

    X = district_data[['Turnout_Rate'] + [col for col in district_data.columns if '_Percentage' in col]]
    X = X.reindex(columns=feature_names, fill_value=0)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    vote_percentage_prediction = reg_model.predict(X_imputed)
    winner_prediction = class_model.predict(X_imputed)

    print(f"Predicted vote percentage for TrumpR_24P_President: {vote_percentage_prediction[0]}")
    print(f"Predicted winner for President in district {district_identifier}: {winner_prediction[0]}")

    if not turnout_data.empty and 'Turnout_Rate' in turnout_data.columns:
        average_turnout_rate = turnout_data['Turnout_Rate'].mean()
        turnout_percentage = average_turnout_rate * 100
    else:
        print("Warning: 'Turnout_Rate' data not available from voter data files. Setting turnout_percentage to 0.")
        turnout_percentage = 0.0

    print(f"Predicted voter turnout rate for district {district_identifier}: {turnout_percentage:.2f}%")

    congressional_vote_percentage = vote_percentage_prediction[0]  
    congressional_winner_prediction = winner_prediction[0] 

    print(f"Predicted vote percentage for Congressional Representative: {congressional_vote_percentage}")
    print(f"Predicted winner for Congressional Representative in district {district_identifier}: {congressional_winner_prediction}")
