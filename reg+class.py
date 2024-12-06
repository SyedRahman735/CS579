import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.impute import SimpleImputer
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

# Create 'Winner_President' column based on the highest vote percentage among candidates
final_data['Winner_President'] = final_data[candidate_columns].idxmax(axis=1)

# Filter out rows where the winner cannot be determined (e.g., all values were NaN)
final_data = final_data[final_data['Winner_President'] != -1]

# Define features (X) and target variables (y) for regression and classification
X = final_data[['Turnout_Rate'] + candidate_columns]
y_regression = final_data['TrumpR_24P_President_Percentage']  # Example target for regression
y_classification = final_data['Winner_President']  # Example target for classification

# Replace Inf and -Inf values with NaN in X
X.replace([float('inf'), -float('inf')], float('nan'), inplace=True)

# Handle missing values using SimpleImputer for regression
imputer = SimpleImputer(strategy='mean')
X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_regression, test_size=0.2, random_state=42)
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Handle missing values using SimpleImputer for classification
X_train_class, X_test_class, y_class_train, y_class_test = train_test_split(X, y_classification, test_size=0.2, random_state=42)
X_train_class = imputer.fit_transform(X_train_class)
X_test_class = imputer.transform(X_test_class)

# Initialize and train the regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_reg_train)

# Save the regression model using joblib
joblib.dump(reg_model, 'regression_model.pkl')

# Predict and evaluate the regression model
y_reg_pred = reg_model.predict(X_test)
regression_mse = mean_squared_error(y_reg_test, y_reg_pred)
print(f"Regression Model MSE: {regression_mse}")

# Initialize and train the classification model
class_model = LogisticRegression(max_iter=1000)
class_model.fit(X_train_class, y_class_train)

# Save the classification model using joblib
joblib.dump(class_model, 'classification_model.pkl')

# Predict and evaluate the classification model
y_class_pred = class_model.predict(X_test_class)
classification_accuracy = accuracy_score(y_class_test, y_class_pred)
print(f"Classification Model Accuracy: {classification_accuracy}")
