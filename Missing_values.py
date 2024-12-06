import pandas as pd
import os

# Define a function to handle missing values in key columns
def handle_missing_values(df):
    # Fill missing values in voter registration with the median
    if 'Voter_Registration' in df.columns:
        df['Voter_Registration'].fillna(df['Voter_Registration'].median(), inplace=True)
    
    # Fill missing values in turnout with the median
    if 'Turnout' in df.columns:
        df['Turnout'].fillna(df['Turnout'].median(), inplace=True)
    
    # Iterate through vote count columns and fill missing values with 0 (assuming no votes cast)
    vote_columns = [col for col in df.columns if 'Votes' in col or 'President' in col or 'Rep' in col]
    for col in vote_columns:
        df[col].fillna(0, inplace=True)
    
    # Drop rows where critical columns like CNTYVTD are missing
    df.dropna(subset=['CNTYVTD', 'VTDKEY'], inplace=True)
    
    return df

# Specify the directory where your cleaned data is stored
base_dir = "Dataset"

# Iterate through each folder and process cleaned CSV files
folders = [
    "2018 Democratic Primary", "2018 General", "2018 Republican Primary",
    "2020 Democratic Primary", "2020 General",
    "2022 Democratic Primary", "2022 General", "2022 Republican Primary",
    "2024 Democratic Primary", "2024 Republican Primary"
]

for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    # List all cleaned CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.startswith('cleaned_') and f.endswith('.csv')]
    
    for csv_file in csv_files:
        # Load each cleaned CSV file into a DataFrame
        file_path = os.path.join(folder_path, csv_file)
        data = pd.read_csv(file_path)
        
        # Handle missing values using the defined function
        cleaned_data = handle_missing_values(data)
        
        # Save the cleaned data with missing values handled
        final_cleaned_file_path = os.path.join(folder_path, f"final_{csv_file}")
        cleaned_data.to_csv(final_cleaned_file_path, index=False)
        
        print(f"Handled missing values and saved: {final_cleaned_file_path}")
