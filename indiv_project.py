"""
Course Number: ENGR 13300
Semester: Fall 2025

Description: Generates synthetic EEG-like signals for three classes ('yes', 'no', 'help'), 
extracts spectral features, trains a KNN classifier to predict the class labels, outputs 
the classification report and time and frequency domain graphs of the signal, and allows
for the prediction of new synthetic or user-provided signals, as well as the option to
save signals and features to CSV files.

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
import matplotlib.pyplot as plt
import csv, os
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from helper_functions import generate_synthetic_signal, add_gaussian_noise, save_signal_to_csv, save_features_csv

# global constants used throughout the program
DEFAULT_SAMPLING_RATE = 256
DEFAULT_DURATION = 1.0
AVAILABLE_LABELS = ['yes', 'no', 'help']
FREQ_MAP = {'yes':6.0,'no':10.0,'help':20.0}
# theta: yes/no
# alpha: higher cognitive loads indicate no
# beta: help (more alertness)

# directories for saving output data and signals
FEATURE_PATH = Path("output_data")
FEATURE_PATH.mkdir(exist_ok=True, parents=True)
SIGNAL_PATH = Path("output_signals")
SIGNAL_PATH.mkdir(exist_ok=True, parents=True)

def prompt_number(msg, i_type, min_val=None, max_val=None):
    while True: # infinite loop until valid input
        try:
            val = i_type(input(msg + " ")) # convert input to desired type
            # checks if within bounds
            if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
                print(f"Value must be between {min_val} and {max_val}")
            else:
                return val
        except ValueError: # if conversion fails
            print("Invalid input. Try again.")

def synthesize_signal(base_freq, sampling_rate, duration, noise_std):
    # generates noisy synthetic EEG signal
    phase = np.random.uniform(0, 2*np.pi) # random phase for variability
    # main, harmonic, and slow drift components
    _, base = generate_synthetic_signal(base_freq, sampling_rate, duration, 1.0, phase) # core signal
    _, harm = generate_synthetic_signal(base_freq*2, sampling_rate, duration, 0.3, phase/2) # smaller harmonic like real EEG
    _, drift = generate_synthetic_signal(0.5, sampling_rate, duration, np.random.uniform(0,0.2), 0) # slow drift component like baseline drift
    return add_gaussian_noise(base + harm + drift, noise_std) # add noise to combined signal

def build_synthetic_dataset(samples_per_class, sampling_rate, duration, noise_std):
    signals, labels = [], [] # lists to hold signals and labels
    t = np.linspace(0, duration, int(sampling_rate*duration), endpoint=False) # time vector
    # loops through each label and generates specified number of samples
    for label in AVAILABLE_LABELS: 
        for i in range(samples_per_class):
            # generate synthetic signal for current label
            signals.append(synthesize_signal(FREQ_MAP[label], sampling_rate, duration, noise_std))
            labels.append(label)
    return np.array(signals), np.array(labels), t

def extract_spectral_features(signals, sampling_rate):
    n_points = signals.shape[1] # number of points in each signal
    freqs = np.fft.rfftfreq(n_points, 1/sampling_rate) # calculates the frequencies corresponding to the output of a real Fast Fourier Transform
    bands = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,60)} # standard EEG bands
    features = []
    
    for sig in signals:
        fft_vals = np.abs(np.fft.rfft(sig))**2 # power spectrum of the signal (distribution of power across frequencies)
        total_power = np.sum(fft_vals)+1e-12 # avoid division by zero
        band_rel = [np.sum(fft_vals[(freqs>=lo)&(freqs<=hi)])/total_power for lo,hi in bands.values()] # relative power in each band
        peak_freq = freqs[np.argmax(fft_vals)] # frequency with maximum power
        centroid = np.sum(freqs*fft_vals)/total_power # spectral centroid (weighted average frequency)
        features.append(band_rel + [peak_freq, centroid])
    
    return np.array(features) # return feature matrix

def train_classifier(X, y, k=5):
    # imported function from sklearn to split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y) # 20% for testing and 80% for training, ensures correct class proportions
    scaler = StandardScaler() # standardize features to unit variance
    X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test) # applying scaling parameters from training set to test set
    
    model = KNeighborsClassifier(n_neighbors=k) # imported function to create KNN model
    model.fit(X_train_scaled, y_train) # train model on training data and labels
    preds = model.predict(X_test_scaled) # predict labels for test set

    return model, scaler, accuracy_score(y_test,preds), classification_report(y_test, preds, zero_division=0)

def plot_signal(signal, t, sr, title="Signal"):
    # plots signal in time domain and magnitude spectrum
    fft_vals = np.abs(np.fft.rfft(signal)) # fft magnitude values
    freqs = np.fft.rfftfreq(len(signal), 1/sr) # corresponding frequencies
    
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1) # time domain plot showing the shape of the signal waveform
    plt.plot(t, signal) # oscillations indicating predictable patterns for classification
    plt.xlabel("Time (s)")
    plt.title("Time :- " + title)
    
    plt.subplot(1,2,2) # frequency domain plot showing magnitude spectrum
    plt.plot(freqs, fft_vals) # peaks indicate dominant frequencies for classification + bands for feature extraction
    plt.xlabel("Freq (Hz)")
    plt.title("Frequency :- " + title)
    plt.tight_layout()
    plt.show()

def main():
    print("-=-=-=- Synthetic BCI Classifier -=-=-=-")
    # user inputs for dataset parameters
    samples = prompt_number("Samples per class (5-500):", int, 5, 500)
    noise = prompt_number("Noise std (0-5):", float, 0, 5)
    sr = prompt_number("Sampling rate (32-2000):", int, 32, 2000)
    dur = prompt_number("Duration (s) (0.1-10):", float, 0.1, 10)
    k = prompt_number("K for KNN (1-31):", int, 1, 31)
    inspect = input("\nInspect example signal? (y/n) ").strip().lower() == 'y'
    n_save = prompt_number("Number of example signals to save (0-Samples):", int, 0, samples) 

    # Build dataset
    signals, labels, t = build_synthetic_dataset(samples, sr, dur, noise)
    print(f"\n[main] Created {len(signals)} signals")

    if n_save > 0:
        for label in AVAILABLE_LABELS: # Save n_save signals per class
            # find indices of signals with this label
            label_indices = np.where(labels == label)[0]

            for j, idx in enumerate(label_indices[:n_save]): # save first n_save signals of this label
                file_path = SIGNAL_PATH / f"signal_{label}_{j}_idx{idx}.csv"
                save_signal_to_csv(str(file_path), t, signals[idx])
        print(f"[main] Saved {n_save} signals per class to '{SIGNAL_PATH}/'")
    
    else:
        print("[main] No signals saved.")

    features = extract_spectral_features(signals, sr) # extract features from all signals
    save_features_csv(str(FEATURE_PATH/"features_labels.csv"), features, labels)

    model, scaler, acc, report = train_classifier(features, labels, k) # train KNN classifier
    
    # report contains performance metrics:
    # Precision: accuracy of positive predictions
    # recall: measures number of positive cases
    # F1-score: weighted average of precision and recall
    # support: number of actual occurrences of each class

    print(f"\nAccuracy: {acc:.3f}\nReport:\n{report}") # print performance

    if inspect: #check if user wants to inspect example signals
        idx = np.random.randint(len(signals)) # random index to inspect
        plot_signal(signals[idx], t, sr, title=f"Example Index {idx} with Label '{labels[idx]}'")
        feat = extract_spectral_features(signals[idx].reshape(1,-1), sr)
        
        pred = model.predict(scaler.transform(feat))[0] # predict label for this signal
        print(f"True label: {labels[idx]}, Predicted: {pred}\n")

    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    print("Interactive prediction mode. Type 'quit' to exit.") # loop for user to test predictions
    while True:
        choice = input("\nType 'file', 'synth', or 'quit': ").strip().lower()
        
        if choice=='quit': # exit the loop and end program
            break
        
        elif choice=='synth': # synthesize new signal for prediction
            label = input(f"Label ({', '.join(AVAILABLE_LABELS)}) or 'random': ").strip().lower()
            if label=='random':
                label = np.random.choice(AVAILABLE_LABELS) # random label if user chooses
            else:
                label = label.lower()
            
            while label not in AVAILABLE_LABELS: 
                label = input("Invalid label. Try again: ").strip().lower() # validate label input
            
            noise_lvl = prompt_number("Noise std for test signal:", float, 0, 5)
            sig = synthesize_signal(FREQ_MAP[label], sr, dur, noise_lvl) # generate signal
            feat = extract_spectral_features(sig.reshape(1,-1), sr)
            pred = model.predict(scaler.transform(feat))[0]
            print(f"Synthesized true: {label}, Predicted: {pred}") # print prediction
            plot_signal(sig, t, sr, title=f"Synth_{label}_pred_{pred}")
        
        elif choice=='file': # load signal from user-provided CSV file
            path = input("Enter CSV path or 'cancel': ").strip()
            
            if path.lower()=='cancel':  # user chose to cancel
                continue
            
            try:
                data = np.loadtxt(path, delimiter=",", skiprows=1) # load CSV data skipping header
                if data.shape[1]>1: # get signal column
                    sig = data[:,1] 
                else:
                    sig = data[:,0] 
                
                if len(sig)<10:  # too short to be valid
                    print("Too short")
                    continue
                
                if len(sig)<len(t): # pads to required length
                    sig = np.pad(sig,(0,len(t)-len(sig)),'constant')
                else: 
                    sig = sig[:len(t)] 

                feat = extract_spectral_features(sig.reshape(1,-1), sr)
                pred = model.predict(scaler.transform(feat))[0]
                print(f"Predicted label: {pred}") # print predicted label
                plot_signal(sig, t, sr, title=f"FilePred_{pred}")
            
            except Exception as e:
                print("Error reading file:", e)
        
        else: 
            print("Invalid option.")

    print("\n~~~~~~~~~~~~Program completed.~~~~~~~~~~~~\n")

if __name__=="__main__":
    main()
