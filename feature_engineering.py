import pandas as pd
import os

# Define a function to perform feature engineering
def feature_engineering(df):
    # Calculate turnout rate as a proportion of registered voters
    if 'Voter_Registration' in df.columns and 'Turnout' in df.columns:
        df['Turnout_Rate'] = df['Turnout'] / df['Voter_Registration']
        df['Turnout_Rate'].fillna(0, inplace=True)  # Handle divisions by zero

    # Calculate vote percentages for each candidate
    vote_columns = [col for col in df.columns if 'President' in col or 'Rep' in col or 'Votes' in col]
    if vote_columns:
        df['Total_Votes'] = df[vote_columns].sum(axis=1)
        for col in vote_columns:
            percentage_col = f"{col}_Percentage"
            df[percentage_col] = df[col] / df['Total_Votes']
            df[percentage_col].fillna(0, inplace=True)  # Handle divisions by zero

        # Drop the intermediate `Total_Votes` column only if it exists
        if 'Total_Votes' in df.columns:
            df.drop(columns=['Total_Votes'], inplace=True)

    return df

# Specify the directory where your consistent data is stored
base_dir = "Dataset"

# Iterate through each folder and process consistent CSV files
folders = [
    "2018 Democratic Primary", "2018 General", "2018 Republican Primary",
    "2020 Democratic Primary", "2020 General",
    "2022 Democratic Primary", "2022 General", "2022 Republican Primary",
    "2024 Democratic Primary", "2024 Republican Primary"
]

for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    # List all consistent CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.startswith('consistent_') and f.endswith('.csv')]
    
    for csv_file in csv_files:
        # Load each consistent CSV file into a DataFrame
        file_path = os.path.join(folder_path, csv_file)
        data = pd.read_csv(file_path)
        
        # Perform feature engineering using the defined function
        engineered_data = feature_engineering(data)
        
        # Save the data with engineered features
        engineered_file_path = os.path.join(folder_path, f"engineered_{csv_file}")
        engineered_data.to_csv(engineered_file_path, index=False)
        
        print(f"Engineered features and saved: {engineered_file_path}")
