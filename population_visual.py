import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Load the county totals data
county_totals = pd.read_csv('Dataset/county_totals.csv')

# Load the Texas shapefile
texas_map = gpd.read_file('Dataset/vtds_24P/VTDs_24P.shp')

# Ensure both 'CNTY' columns are treated as integers for compatibility
county_totals['CNTY'] = county_totals['CNTY'].astype(int)
texas_map['CNTY'] = texas_map['CNTY'].astype(int)

# Merge the shapefile with the county totals on 'CNTY'
merged_data = texas_map.merge(county_totals, on='CNTY', how='left')

# Apply a logarithmic transformation to the 'Total' population column
merged_data['Total_log'] = np.log1p(merged_data['Total'])  # log1p for log(1 + x) to handle zeros

# Plot the map with the 'Viridis' colormap and only one formatted color bar
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
plot = merged_data.plot(column='Total_log', cmap='viridis', linewidth=0.5, ax=ax, edgecolor='0.5', legend=False)
ax.set_title("Texas Population by County")
ax.set_axis_off()

# Add a single color bar and format it with population values
colorbar = fig.colorbar(plot.get_children()[0], ax=ax)
colorbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(np.expm1(x)):,}'))  # expm1 to reverse log1p

plt.show()
