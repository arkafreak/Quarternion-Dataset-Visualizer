#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# In[31]:


# Load the CSV file into a DataFrame
data = pd.read_csv("rocket_orientation_data.csv")

# Drop a specific column by specifying its name
column_to_drop = 'Timestamp'
cleaned_data = data.drop(columns=[column_to_drop])

# Save the cleaned DataFrame back to a CSV file
cleaned_data.to_csv("cleaned_data.csv", index=False)


# In[32]:


numeric_columns = cleaned_data.select_dtypes(include=[np.number])

scaler = MinMaxScaler()

normalized_data = scaler.fit_transform(numeric_columns)

normalized_df = pd.DataFrame(normalized_data, columns=numeric_columns.columns)

print(normalized_df)


# In[33]:



from sklearn.preprocessing import StandardScaler

numeric_columns = cleaned_data.select_dtypes(include=[np.number])

scaler = StandardScaler()

standardized_data = scaler.fit_transform(numeric_columns)

data[numeric_columns.columns] = standardized_data

print(data)


# In[36]:


column_names = data.columns
sg_data = data[:10]
# Determine the number of rows and columns for subplots
num_cols = len(column_names)
num_rows = 1

# Set up the subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(35, 5))

# Loop through each column and create a subplot
for i, column in enumerate(column_names):
    ax = axes[i]
    sg_data[column].plot(ax=ax)
    ax.set_title(column)
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.grid()

plt.tight_layout()
plt.show()


# In[37]:


#sg_data = data[:10]
sg_data['diff_qc1_qc0'] = sg_data['Qc1'] - sg_data['Qc0']
sg_data['diff_qc2_qc0'] = sg_data['Qc2'] - sg_data['Qc0']
sg_data['diff_qc3_qc0'] = sg_data['Qc3'] - sg_data['Qc0']
sg_data['diff_qc2_qc1'] = sg_data['Qc2'] - sg_data['Qc1']
sg_data['diff_qc3_qc1'] = sg_data['Qc3'] - sg_data['Qc1']
sg_data['diff_qc3_qc2'] = sg_data['Qc3'] - sg_data['Qc2']

# Plot the difference between 'Qc1' and 'Qc0'
plt.figure(figsize=(10, 6))
plt.plot(sg_data['Timestamp'], sg_data['diff_qc1_qc0'], label='Qc1 - Qc0')
plt.title('Difference between Qc0 and Qc1 Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Difference')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot the difference between 'Qc2' and 'Qc0'
plt.figure(figsize=(10, 6))
plt.plot(sg_data['Timestamp'], sg_data['diff_qc2_qc0'], label='Qc2 - Qc0')
plt.title('Difference between Qc0 and Qc2 Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Difference')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot the difference between 'Qc3' and 'Qc0'
plt.figure(figsize=(10, 6))
plt.plot(sg_data['Timestamp'], sg_data['diff_qc3_qc0'], label='Qc3 - Qc0')
plt.title('Difference between Qc0 and Qc3 Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Difference')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot the difference between 'Qc2' and 'Qc1'
plt.figure(figsize=(10, 6))
plt.plot(sg_data['Timestamp'], sg_data['diff_qc2_qc1'], label='Qc2 - Qc1')
plt.title('Difference between Qc1 and Qc2 Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Difference')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot the difference between 'Qc3' and 'Qc1'
plt.figure(figsize=(10, 6))
plt.plot(sg_data['Timestamp'], sg_data['diff_qc3_qc1'], label='Qc3 - Qc1')
plt.title('Difference between Qc1 and Qc3 Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Difference')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot the difference between 'Qc3' and 'Qc2'
plt.figure(figsize=(10, 6))
plt.plot(sg_data['Timestamp'], sg_data['diff_qc3_qc2'], label='Qc3 - Qc2')
plt.title('Difference between Qc2 and Qc3 Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Difference')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# In[ ]:




