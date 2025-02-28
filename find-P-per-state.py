import pandas as pd
import numpy as np
import os

# Define folder paths
desktop_folder = os.path.expanduser('~/Desktop/state')
output_folder = os.path.expanduser('~/Desktop/graph')

# Create output folder if not exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Dictionary to store results for each subsub folder
all_results = {}

# Traverse each sub-folder in the main folder
for subfolder in os.listdir(desktop_folder):
    subfolder_path = os.path.join(desktop_folder, subfolder)
    if os.path.isdir(subfolder_path):
        
        # Dictionary to store probabilities for each rat across n=2 to n=6
        rat_probabilities = {}
        
        # Traverse each subsub-folder (n=2 to n=6)
        for subsubfolder in sorted(os.listdir(subfolder_path)):
            subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
            if os.path.isdir(subsubfolder_path):
                
                # Process each CSV file in the subsubfolder
                for file in os.listdir(subsubfolder_path):
                    if file.startswith("Trial_arrays_Ratrat") and file.endswith(".csv"):
                        file_path = os.path.join(subsubfolder_path, file)
                        
                        try:
                            # Read CSV file
                            df_trial_sample = pd.read_csv(file_path, encoding="ISO-8859-1")
                            
                            # Extract relevant columns
                            w_list_cols_sorted = sorted([col for col in df_trial_sample.columns if "w_list" in col])
                            nLL_list_cols_sorted = sorted([col for col in df_trial_sample.columns if "nLL_list" in col])
                            
                            # Ensure column count matches
                            if len(w_list_cols_sorted) == len(nLL_list_cols_sorted):
                                # Compute trial-level NLL: sum(w_list * nLL_list) per trial
                                trial_nlls = sum(df_trial_sample[w] * df_trial_sample[n] for w, n in zip(w_list_cols_sorted, nLL_list_cols_sorted))
                                
                                # Compute overall NLL
                                overall_nll = trial_nlls.sum()
                                
                                # Compute average NLL per trial
                                avg_nll = overall_nll / len(trial_nlls) if len(trial_nlls) > 0 else np.nan
                                
                                # Compute probability
                                probability = np.exp(-avg_nll) if not np.isnan(avg_nll) else np.nan
                                
                                # Extract rat ID from filename
                                rat_id = file.split("Trial_arrays_Ratrat")[-1].split(".csv")[0]
                                
                                # Store probability for this rat and n group
                                if rat_id not in rat_probabilities:
                                    rat_probabilities[rat_id] = {}
                                rat_probabilities[rat_id][subsubfolder] = probability
                        
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")
                
        # Convert results to DataFrame
        df_nll_results = pd.DataFrame.from_dict(rat_probabilities, orient="index")
        df_nll_results.reset_index(inplace=True)
        df_nll_results.rename(columns={'index': 'Rat ID'}, inplace=True)
        
        # Save results to CSV in the output folder
        output_file = os.path.join(output_folder, f"{subfolder}_probabilities.csv")
        df_nll_results.to_csv(output_file, index=False)
        
        # Store results for overall processing
        all_results[subfolder] = df_nll_results
