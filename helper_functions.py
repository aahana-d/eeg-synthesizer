"""
Course Number: ENGR 13300
Semester: Fall 2025

Description: This module provides helper functions for the Decoding Intent BCI Simulator Project
for creating and saving synthetic EEG-like signals.

Assignment Information:
    Assignment:     Individual Project
    Team ID:        LC2 - 20
    Author:         Aahana Dahiya, dahiya1@purdue.edu
    Date:           12/4/2025

Academic Integrity Statement:
    I have not used source code obtained from any unauthorized
    source, either modified or unmodified; nor have I provided
    another student access to my code.  The project I am
    submitting is my own original work.
"""

import numpy as np
import csv
from pathlib import Path

def generate_synthetic_signal(freq, sampling, duration, amp=1.0, phase=0.0):
    # inclusion of optional inputs for amplitude and phase to allow more flexible signal generation
    num_samples = int(sampling * duration) # total number of data points
    t = np.linspace(0, duration, num_samples, endpoint = False) # generates time vector, avoid double-counting endpoint
    signal = amp * np.sin(2 * np.pi * freq * t + phase) # create a simple sinusoid
    # Returns time vector and signal as numpy arrays
    return t, signal

def add_gaussian_noise(signal_array, noise_std): # generates noise from normal distribution to model environmental interference
    noise = np.random.normal(loc = 0.0, scale = noise_std, size = signal_array.shape) 
    #artificially corrupt a given dataset by adding random fluctuation
    return signal_array + noise

def save_signal_to_csv(filename, time_vector, signal_vector, header="time, signal"):
    try: # attempts to save the signal to a CSV file
        filepath = Path(filename)
        if not filepath.parent.exists(): # checks if given file exists, otherwise creates the parent directories
            filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', newline='') as csvfile: 
            # saves the time and signal values in csv format
            writer = csv.writer(csvfile)
            writer.writerow(header.split(', ')) # writes the header row
            for i in range(len(time_vector)): # iterates through each data point
                t = time_vector[i]
                s = signal_vector[i]
                writer.writerow([f"{t:.6f}", f"{s:.6f}"])
        return True # indicates successful save
    
    except Exception as e: # any error occurred during the save process
        print(f"[helper_functions.save_signal_to_csv] Error while saving file: {e}") # prints error message
        # returns False to allow reattempt rather than crashing
        return False
    
def save_features_csv(filename, features, labels):
    header = ['delta','theta','alpha','beta','gamma','peak_freq','centroid','label'] # column names
    try: # attempts to save features and labels to CSV
        with open(filename, 'w', newline='') as f:
            f.write(', '.join(header) + '\n')
            for feat, lab in zip(features, labels): # aggregates features and labels
                f.write(', '.join(f"{v:.6f}" for v in feat) + f",{lab}\n") # writes each feature vector and label
        print(f"[save_features_csv] Saved features to {filename}")
        return True # indicates successful save
    
    except Exception as e: # any error occurred during the save process
        print(f"[save_features_csv] Error: {e}")
        return False # indicates failure to save