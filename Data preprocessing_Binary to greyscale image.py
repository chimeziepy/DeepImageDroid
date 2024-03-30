#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


# In[2]:


# Read the dataset from the CSV file
#df = pd.read_csv("NAticusdriod_1.csv")
#df = pd.read_csv("NAticusdriod_0.csv")
#df = pd.read_csv("drebin-215-dataset_B.csv")
#df = pd.read_csv("andriodHealthCheck_B.csv")

df = pd.read_csv("drebin-215-dataset_s.csv")

# Create a folder to save the images if it doesn't exist

output_folder = 'drebinGreyscale_215_dataset_S'
os.makedirs(output_folder, exist_ok=True)

def save_image(index, data):
    # Normalize pixel values to be in the range [0, 1]
    normalized_data = data / 255.0
    
    # Resize the image to a fixed size (e.g., 64x64)
    resized_data = cv2.resize(normalized_data, (64, 64))
    
    # Create a grayscale image
    plt.imshow(resized_data, cmap='gray', interpolation='none', aspect='auto')
    plt.axis('off')
    
    # Save the image to the output folder
    filename = os.path.join(output_folder, f'row_{index}.png')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, format='png', dpi=300)
    plt.close()

# Define a function to process a batch of rows
def process_batch(batch):
    for index, row in batch.iterrows():
        row_data = row.values.reshape((1, len(row)))
        save_image(index, row_data)

# Set the number of threads based on your system's capabilities
num_threads = 4

# Split the DataFrame into batches
batch_size = len(df) // num_threads
df_batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]

# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    executor.map(process_batch, df_batches)

print("Grayscale images for each row have been saved in the folder:", output_folder)


# In[12]:


# Read the dataset from the CSV file
df = pd.read_csv("andriodHealthCheck_S.csv", low_memory=False)

# Create a folder to save the images if it doesn't exist
output_folder = 'grayscale_images_andriodHealthCheck_s'
os.makedirs(output_folder, exist_ok=True)

def save_image(row, index, filename):
    # Extract numeric values from the row
    numeric_values = pd.to_numeric(row, errors='coerce')
    
    # Check if all values are numeric
    if numeric_values.notnull().all():
        # Normalize pixel values to be in the range [0, 1]
        normalized_data = numeric_values.astype(float) / 255.0
        
        # Resize the image to a fixed size (e.g., 64x64)
        resized_data = cv2.resize(normalized_data.values.reshape(1, -1), (64, 64))
        
        # Create a grayscale image
        plt.imshow(resized_data, cmap='gray', interpolation='none', aspect='auto')
        plt.axis('off')
        
        # Save the image to the output folder
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, format='png', dpi=300)
        plt.close()
    else:
        print(f"Skipping row {index} due to non-numeric values.")

# Process and save images for each row
for index, row in df.iterrows():
    save_image(row, index, os.path.join(output_folder, f'row_{index}.png'))

print("Grayscale images for each row have been saved in the folder:", output_folder)

