import pandas as pd

# Load the data
population_data = pd.read_csv('Dataset/demographics.csv')

# Sum the 'total' values for each unique 'CNTY' and save as a new DataFrame
county_totals = population_data.groupby('CNTY')['total'].sum().reset_index()

# Rename columns to match the specified format
county_totals.columns = ['CNTY', 'Total']

# Save the result to a new CSV file
county_totals.to_csv('county_totals.csv', index=False)

print("Summed totals for each county have been saved to 'county_totals.csv'")
