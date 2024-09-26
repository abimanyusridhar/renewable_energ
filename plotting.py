# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "C:/Users/vidya/Downloads/daily-renewable-energy-generation.csv"
data = pd.read_csv(file_path)

# Convert date column to datetime
data['date'] = pd.to_datetime(data['date'])

# Basic data preprocessing
data.dropna(subset=['actual_generation', 'installed_capacity'], inplace=True)

df_filtered = data[data['station'].isin(['DADRI SOLAR','SINGRAULI SOLAR','TATA RENEWABLES','YARROW','TATA POWER','SBG ENERGY','RENEW SOLAR (BHUVAD)','OSTRO WIND (KUTCH)','ACME SOLAR (RAMNAGAR)','MAHINDRA SOLAR (BADWAR)','AZURE POWER INDIA PVT LTD','TATA POWER RENEWABLE ENERGY LTD','GREEN INFRA','MYTRA'])]
df_filtered.shape
data=df_filtered

# Convert categorical variables to string for plotting
data['state_name'] = data['state_name'].astype(str)
data['station'] = data['station'].astype(str)
data['type'] = data['type'].astype(str)

# Set the plot style
sns.set(style="whitegrid")

# 1. Plot actual generation vs state_name
plt.figure(figsize=(10, 6))
sns.barplot(x='state_name', y='actual_generation', data=data)
plt.title('Actual Generation vs State Name', fontsize=16)
plt.xticks(rotation=45)
plt.ylabel('Actual Generation (MW)')
plt.xlabel('State Name')
plt.tight_layout()
plt.show()

# 2. Plot actual generation vs installed_capacity
plt.figure(figsize=(10, 6))
sns.barplot(x='installed_capacity', y='actual_generation', data=data, hue='type')
plt.title('Actual Generation vs Installed Capacity', fontsize=16)
plt.ylabel('Actual Generation (MW)')
plt.xlabel('Installed Capacity (MW)')
plt.legend(title='Type', loc='upper right')
plt.tight_layout()
plt.show()

# 3. Plot actual generation vs station
plt.figure(figsize=(12, 6))
sns.barplot(x='station', y='actual_generation', data=data)
plt.title('Actual Generation vs Station', fontsize=16)
plt.xticks(rotation=90)
plt.ylabel('Actual Generation (MW)')
plt.xlabel('Station Name')
plt.tight_layout()
plt.show()

# 4. Plot actual generation vs type
plt.figure(figsize=(8, 6))
sns.barplot(x='type', y='actual_generation', data=data)
plt.title('Actual Generation vs Type', fontsize=16)
plt.ylabel('Actual Generation (MW)')
plt.xlabel('Energy Type')
plt.tight_layout()
plt.show()
