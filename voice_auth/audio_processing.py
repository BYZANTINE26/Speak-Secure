import numpy as np
import python_speech_features as mfcc
from sklearn import preprocessing


def calculate_delta(array):
    N = 2
    rows, cols = array.shape

    # Create an array of indices for the sliding window
    indices = np.arange(-N, N + 1)
    
    # Ensure the indices stay within bounds
    indices = np.clip(indices + np.arange(rows)[:, np.newaxis], 0, rows - 1)

    # Calculate the delta using array slicing and operations
    left_neighbors = array[indices[:, N - 1]]
    right_neighbors = array[indices[:, N + 1]]
    delta = (right_neighbors - left_neighbors) + 2 * (right_neighbors - 2 * array[indices[:, N]] + left_neighbors)
    delta /= 10

    return delta

def extract_features(audio, rate):
    mfcc_features = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, appendEnergy=True, nfft=1103)
    mfcc_features = preprocessing.scale(mfcc_features)
    delta = calculate_delta(mfcc_features)
    combined = np.hstack((mfcc_features, delta))
    return combined