import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Define folder paths
output_folder = os.path.expanduser('~/Desktop/graph')

# List all probability CSV files
probability_files = [f for f in os.listdir(output_folder) if f.endswith("_probabilities.csv")]

# Create a total results list for aggregated plotting
total_data = []

# Process each probability CSV file
for file in probability_files:
    file_path = os.path.join(output_folder, file)
    df = pd.read_csv(file_path)
    
    # Plot individual subfolder data
    plt.figure(figsize=(8, 6))
    for _, row in df.iterrows():
        probabilities = row[1:].values  # Extract n=2 to n=6 probabilities
        x_values = list(range(2, 2 + len(probabilities)))
        plt.plot(x_values, probabilities, marker='o', linestyle='-')
    
    plt.xlabel("n-state")
    plt.ylabel("Probability")
    plt.title(f"Probability Progression - {file}")
    plt.grid()
    output_img_path = os.path.join(output_folder, file.replace(".csv", ".png"))
    plt.savefig(output_img_path)
    plt.close()
    
    # Store for total data
    total_data.append(df.set_index("Rat ID"))

# Combine all data for total plot
total_df = pd.concat(total_data, axis=0)
plt.figure(figsize=(10, 6))
for _, row in total_df.iterrows():
    probabilities = row.values
    x_values = list(range(2, 2 + len(probabilities)))
    plt.plot(x_values, probabilities, marker='o', linestyle='-')

plt.xlabel("n-state")
plt.ylabel("Probability")
plt.title("Overall Probability Progression")
plt.grid()
total_output_path = os.path.join(output_folder, "Total_Probability_Progression.png")
plt.savefig(total_output_path)
plt.close()
