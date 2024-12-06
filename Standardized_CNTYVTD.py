import pandas as pd
import os

# Define the cleaning function for CNTYVTD
def clean_cntyvtd(value):
    return ''.join(filter(str.isdigit, str(value)))

# Specify the directory where your folders are stored
base_dir = "Dataset"

# Iterate through each folder and process CSV files
folders = [
    "2018 Democratic Primary", "2018 General", "2018 Republican Primary",
    "2020 Democratic Primary", "2020 General",
    "2022 Democratic Primary", "2022 General", "2022 Republican Primary",
    "2024 Democratic Primary", "2024 Republican Primary"
]

for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        # Load each CSV file into a DataFrame
        file_path = os.path.join(folder_path, csv_file)
        data = pd.read_csv(file_path)
        
        # Clean the CNTYVTD column
        data['CNTYVTD'] = data['CNTYVTD'].apply(clean_cntyvtd)
        
        # Save the cleaned data with a new filename
        cleaned_file_path = os.path.join(folder_path, f"cleaned_{csv_file}")
        data.to_csv(cleaned_file_path, index=False)
        
        print(f"Cleaned and saved: {cleaned_file_path}")
