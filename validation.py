import pandas as pd
import os
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score
import numpy as np

# Specify the base directory where the data is stored
base_dir = "Dataset"

# List of folders to iterate over
folders = [
    "2018 Democratic Primary", "2018 General", "2018 Republican Primary",
    "2020 Democratic Primary", "2020 General", "2022 Democratic Primary",
    "2022 General", "2022 Republican Primary",
    "2024 Democratic Primary", "2024 Republican Primary"
]

# Define lists to store the final combined data for cross-validation
combined_data = []

# Iterate through each folder and process only the CSV files starting with "engineered_"
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    csv_files = [f for f in os.listdir(folder_path) if f.startswith('engineered_') and f.endswith('.csv')]
    
    folder_data = []
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        data = pd.read_csv(file_path)
        folder_data.append(data)
    
    if folder_data:
        combined_df = pd.concat(folder_data, ignore_index=True)
        combined_data.append(combined_df)

# Combine all data into a single DataFrame
final_data = pd.concat(combined_data, ignore_index=True)

# Feature and target preparation
candidate_columns = [col for col in final_data.columns if '_Percentage' in col]
print(f"Candidate columns: {candidate_columns}")

# Fill missing values in candidate columns before determining the winner
final_data[candidate_columns] = final_data[candidate_columns].fillna(-1)

# Create the 'Winner_President' column
if candidate_columns:
    final_data['Winner_President'] = final_data[candidate_columns].idxmax(axis=1)
else:
    print("No candidate columns found for creating 'Winner_President'.")

# Define features (X) and target variables (y) for regression and classification
X = final_data[['Turnout_Rate'] + candidate_columns]
y_regression = final_data['TrumpR_24P_President_Percentage']
y_classification = final_data['Winner_President']

# Replace inf values with NaN, and handle missing values
X.replace([float('inf'), -float('inf')], np.nan, inplace=True)
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Define models for cross-validation
reg_model = LinearRegression()
class_model = LogisticRegression(max_iter=1000, class_weight='balanced')

# Cross-validation for regression
mse_scorer = make_scorer(mean_squared_error)
reg_cv_scores = cross_val_score(reg_model, X, y_regression, cv=5, scoring=mse_scorer)
print(f"Regression Cross-Validation MSE Scores: {reg_cv_scores}")
print(f"Mean MSE: {np.mean(reg_cv_scores)}")

# Cross-validation for classification
class_cv_scores = cross_val_score(class_model, X, y_classification, cv=5, scoring='accuracy')
print(f"Classification Cross-Validation Accuracy Scores: {class_cv_scores}")
print(f"Mean Accuracy: {np.mean(class_cv_scores)}")
