# Decoding Intent BCI Simulator

A Python-based Brain-Computer Interface (BCI) simulator that generates synthetic EEG-like signals, extracts spectral features using the Fast Fourier Transform (FFT), and classifies user intent with a K-Nearest Neighbors (KNN) machine learning model. The project provides an interactive interface for generating, visualizing, classifying, and exporting synthetic brainwave data.

---

## Overview

Brain-Computer Interfaces enable communication between the human brain and external systems by interpreting neural activity. Because collecting real EEG datasets requires specialized equipment, this project simulates EEG-like signals corresponding to three simple intent classes:

- **Yes**
- **No**
- **Help**

The generated signals incorporate realistic characteristics such as harmonic components, baseline drift, random phase variation, and Gaussian noise. Spectral features are extracted using the Fast Fourier Transform (FFT) and used to train a K-Nearest Neighbors (KNN) classifier capable of predicting the intended class of new signals.

The simulator also allows users to visualize signals, classify external CSV files, and export generated datasets for future analysis.

---

## Features

- Generate synthetic EEG-like signals for three intent classes
- Simulate realistic signal noise and variability
- Extract spectral features using FFT
- Compute relative EEG band powers:
  - Delta (0.5–4 Hz)
  - Theta (4–8 Hz)
  - Alpha (8–13 Hz)
  - Beta (13–30 Hz)
  - Gamma (30–60 Hz)
- Calculate peak frequency and spectral centroid
- Train and evaluate a K-Nearest Neighbors classifier
- Display classification accuracy and performance metrics
- Visualize signals in both the time and frequency domains
- Save generated signals to CSV files
- Save extracted feature datasets to CSV
- Predict user intent from newly synthesized or imported signals

---

## Technologies Used

- Python 3
- NumPy
- Matplotlib
- scikit-learn
- pathlib
- csv

---

## Project Structure

```
Decoding-Intent-BCI/
│
├── decode_bci.py              # Main application
├── helper_functions.py        # Utility functions
├── output_data/               # Generated feature CSVs
├── output_signals/            # Saved synthetic signals
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<username>/Decoding-Intent-BCI.git
cd Decoding-Intent-BCI
```

Install the required dependencies:

```bash
pip install numpy matplotlib scikit-learn
```

---

## Running the Program

Run:

```bash
python decode_bci.py
```

The program will prompt for:

- Samples per class
- Noise standard deviation
- Sampling rate
- Signal duration
- Whether to inspect an example signal
- Number of signals to save
- K value for the KNN classifier

After training, the program enters an interactive mode where users can:

- Generate new synthetic signals
- Import signals from CSV files
- Predict intent labels
- Visualize signals and spectra
- Exit the program

---

## Example Workflow

1. Generate a synthetic dataset.
2. Extract spectral features.
3. Train the KNN classifier.
4. Evaluate classification accuracy.
5. Visualize an example signal.
6. Generate or load additional signals for prediction.
7. Save signals and feature datasets as CSV files.

---

## Example Output

The program produces:

- Classification accuracy
- Classification report
- Time-domain waveform plots
- Frequency-domain FFT plots
- Saved signal CSV files
- Saved feature CSV datasets

---

## Learning Objectives Demonstrated

- Digital signal processing using FFT
- Synthetic signal generation
- Machine learning with KNN
- Feature extraction
- Data visualization
- File input/output
- Error handling
- Modular programming with user-defined functions

---

## Future Improvements

Possible future enhancements include:

- Support for real EEG datasets
- Additional machine learning models (SVM, Random Forest, Neural Networks)
- Graphical User Interface (GUI)
- Live EEG device integration
- Real-time signal classification
- Expanded intent vocabulary
- More advanced EEG preprocessing and filtering

---

## Author

**Aahana Dahiya**

ENGR 13300 Fall 2025 Individual Project  
Purdue University

---

## License

This project is intended for educational and research purposes.
