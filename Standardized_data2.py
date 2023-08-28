#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt


# In[7]:


data = pd.read_csv("rocket_orientation_data.csv")


# In[8]:


data.head()


# In[13]:


column_to_drop = 'Timestamp'
cleaned_data = data.drop(columns=[column_to_drop])


# In[16]:


cleaned_data.to_csv("cleaned_data.csv", index=False)


# In[18]:


z_score_scaler = StandardScaler()
z_score_normalized = z_score_scaler.fit_transform(data)
print("Z-Score Normalized:")
print(z_score_normalized)


# In[19]:


from sklearn.preprocessing import StandardScaler

numeric_columns = cleaned_data.select_dtypes(include=[np.number])

scaler = StandardScaler()

standardized_data = scaler.fit_transform(numeric_columns)

data[numeric_columns.columns] = standardized_data

print(data)


# In[20]:


sg_data = data[:10]
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


# In[21]:


data['Qc0']


# In[22]:


standard_deviation_Qc0=data['Qc0'].std()


# In[24]:


standard_deviation_Qc1=data['Qc1'].std()


# In[25]:


standard_deviation_Qc2=data['Qc2'].std()


# In[31]:


standard_deviation_Qc3=data['Qc3'].std()


# # Standard Deviation of Qc0

# In[32]:


standard_deviation_Qc0


# # Standard Deviation of Qc1

# In[33]:


standard_deviation_Qc1


# # Standard Deviation of Qc2

# In[34]:


standard_deviation_Qc2


# # Standard Deviation of Qc3

# In[35]:


standard_deviation_Qc3


# # Standard Deviation of the difference between Qc1 and Qc0

# In[37]:


# Calculate the difference between Qc1 and Qc0
difference = np.array(data['Qc1']) - np.array(data['Qc0'])

# Calculate the standard deviation of the difference
std_deviation_difference = np.std(difference)

print("Standard Deviation of the Difference:", std_deviation_difference)


# # Standard Deviation of the difference between Qc2 and Qc0

# In[38]:


# Calculate the difference between Qc2 and Qc0
difference = np.array(data['Qc2']) - np.array(data['Qc0'])

# Calculate the standard deviation of the difference
std_deviation_difference = np.std(difference)

print("Standard Deviation of the Difference:", std_deviation_difference)


# # Standard Deviation of the difference between Qc3 and Qc0

# In[39]:


# Calculate the difference between Qc3 and Qc0
difference = np.array(data['Qc3']) - np.array(data['Qc0'])

# Calculate the standard deviation of the difference
std_deviation_difference = np.std(difference)

print("Standard Deviation of the Difference:", std_deviation_difference)


# # Standard Deviation of the difference between Qc2 and Qc1

# In[40]:


# Calculate the difference between Qc2 and Qc1
difference = np.array(data['Qc2']) - np.array(data['Qc1'])

# Calculate the standard deviation of the difference
std_deviation_difference = np.std(difference)

print("Standard Deviation of the Difference:", std_deviation_difference)


# # Standard Deviation of the difference between Qc3 and Qc1

# In[43]:


# Calculate the difference between Qc3 and Qc1
difference = np.array(data['Qc3']) - np.array(data['Qc1'])

# Calculate the standard deviation of the difference
std_deviation_difference = np.std(difference)

print("Standard Deviation of the Difference:", std_deviation_difference)


# # Standard Deviation of the difference between Qc3 and Qc2

# In[44]:


# Calculate the difference between Qc3 and Qc2
difference = np.array(data['Qc3']) - np.array(data['Qc2'])

# Calculate the standard deviation of the difference
std_deviation_difference = np.std(difference)

print("Standard Deviation of the Difference:", std_deviation_difference)


# In[ ]:




