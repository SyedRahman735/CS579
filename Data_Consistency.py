import pandas as pd
import os

# Define a function to ensure data type consistency
def ensure_data_type_consistency(df):
    # Convert columns to numeric where appropriate
    numeric_columns = ['Voter_Registration', 'Turnout'] + [col for col in df.columns if 'Votes' in col or 'President' in col or 'Rep' in col]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert CNTYVTD and VTDKEY to string to ensure consistent merging
    if 'CNTYVTD' in df.columns:
        df['CNTYVTD'] = df['CNTYVTD'].astype(str)
    if 'VTDKEY' in df.columns:
        df['VTDKEY'] = df['VTDKEY'].astype(str)

    return df

# Specify the directory where your final cleaned data is stored
base_dir = "Dataset"

# Iterate through each folder and process final cleaned CSV files
folders = [
    "2018 Democratic Primary", "2018 General", "2018 Republican Primary",
    "2020 Democratic Primary", "2020 General",
    "2022 Democratic Primary", "2022 General", "2022 Republican Primary",
    "2024 Democratic Primary", "2024 Republican Primary"
]

for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    # List all final cleaned CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.startswith('final_') and f.endswith('.csv')]
    
    for csv_file in csv_files:
        # Load each final cleaned CSV file into a DataFrame
        file_path = os.path.join(folder_path, csv_file)
        data = pd.read_csv(file_path)
        
        # Ensure data type consistency using the defined function
        consistent_data = ensure_data_type_consistency(data)
        
        # Save the data with consistent types
        consistent_file_path = os.path.join(folder_path, f"consistent_{csv_file}")
        consistent_data.to_csv(consistent_file_path, index=False)
        
        print(f"Ensured data type consistency and saved: {consistent_file_path}")
