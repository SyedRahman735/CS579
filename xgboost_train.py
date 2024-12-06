import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb
import joblib

# Specify the base directory where the data is stored
base_dir = "Dataset"

# List of folders to iterate over
folders = [
    "2018 Democratic Primary", "2018 General", "2018 Republican Primary",
    "2020 Democratic Primary", "2020 General", "2022 Democratic Primary",
    "2022 General", "2022 Republican Primary",
    "2024 Democratic Primary", "2024 Republican Primary"
]

# Define lists to store the final combined data for regression and classification
combined_data = []

# Iterate through each folder and process only the CSV files starting with "engineered_"
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    csv_files = [f for f in os.listdir(folder_path) if f.startswith('engineered_') and f.endswith('.csv')]
    
    # Initialize an empty list to store data from each "engineered_" CSV
    folder_data = []

    for csv_file in csv_files:
        # Load each "engineered_" CSV file into a DataFrame
        file_path = os.path.join(folder_path, csv_file)
        data = pd.read_csv(file_path)
        folder_data.append(data)
    
    # Combine all DataFrames in the folder into a single DataFrame
    if folder_data:
        combined_df = pd.concat(folder_data, ignore_index=True)
        
        # Append the combined data for later use in model training
        combined_data.append(combined_df)

# Combine all data from different folders into a single DataFrame for model training
final_data = pd.concat(combined_data, ignore_index=True)

# Define candidate percentage columns for determining the winner
candidate_columns = [col for col in final_data.columns if '_Percentage' in col]

# Fill NaN values with zeros in candidate percentage columns before using idxmax
final_data[candidate_columns] = final_data[candidate_columns].fillna(0)

# Defragment the DataFrame before creating 'Winner_President' column
final_data = final_data.copy()
final_data['Winner_President'] = final_data[candidate_columns].idxmax(axis=1)

# Filter out rows where the winner cannot be determined (e.g., all values were NaN)
final_data = final_data[final_data['Winner_President'] != -1]

# Define features (X) and target variables (y) for regression and classification
X = final_data[['Turnout_Rate'] + candidate_columns]
y_regression = final_data['TrumpR_24P_President_Percentage']  # Example target for regression
y_classification = final_data['Winner_President']  # Example target for classification

# Replace Inf and -Inf values with NaN in X
X = X.replace([float('inf'), -float('inf')], float('nan'))

# Ensure continuous class encoding
y_class_encoded, class_mapping = pd.factorize(final_data['Winner_President'])
print("Class Mapping (Original to Encoded):", dict(enumerate(class_mapping)))

# Print class mapping for debugging
print("Original Classes:", y_classification.unique())
print("Factorized Classes Mapping:", dict(enumerate(class_mapping)))

# Train-test split for regression
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Initialize and train the regression model using XGBoost
reg_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)
reg_model.fit(X_train, y_reg_train)

# Save the regression model using joblib
joblib.dump(reg_model, 'regression_model_xgb.pkl')

# Predict and evaluate the regression model
y_reg_pred = reg_model.predict(X_test)
regression_mse = mean_squared_error(y_reg_test, y_reg_pred)
print(f"Regression Model MSE (XGBoost): {regression_mse}")

# Train-test split for classification
X_train_class, X_test_class, y_class_train, y_class_test = train_test_split(
    X, y_class_encoded, test_size=0.2, random_state=42
)

# Print the unique class labels in the training set
print("Unique Classes in y_class_train:", set(y_class_train))
print("Unique Classes in y_class_test:", set(y_class_test))

import numpy as np

# Ensure all possible classes are present in y_class_train
expected_classes = np.arange(len(class_mapping))  # Continuous integers from 0 to max class
y_class_train = np.concatenate([y_class_train, expected_classes])
X_train_class = np.concatenate([X_train_class, X_train_class[:len(expected_classes)]])


# Impute missing values for classification
X_train_class = imputer.fit_transform(X_train_class)
X_test_class = imputer.transform(X_test_class)

# Initialize and train the classification model using XGBoost
class_model = xgb.XGBClassifier(objective='multi:softmax', n_estimators=100, learning_rate=0.1, max_depth=6)
class_model.fit(X_train_class, y_class_train)

# Save the classification model using joblib
joblib.dump(class_model, 'classification_model_xgb.pkl')

# Predict and evaluate the classification model
y_class_pred = class_model.predict(X_test_class)
classification_accuracy = accuracy_score(y_class_test, y_class_pred)
print(f"Classification Model Accuracy (XGBoost): {classification_accuracy}")
