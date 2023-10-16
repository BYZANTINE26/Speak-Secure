import scipy.io.wavfile as wav
import numpy as np
from speechpy.feature import mfcc

MEAN_SIGNAL_LENGTH = 88500

def get_feature_vector_from_mfcc(file_path: str, flatten: bool, mfcc_len: int = 39) -> np.ndarray:
    fs, signal = wav.read(file_path)
    s_len = len(signal)
    # pad the signals to have same size if lesser than required
    # else slice them
    if s_len < MEAN_SIGNAL_LENGTH:
        pad_len = MEAN_SIGNAL_LENGTH - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values=0)
    else:
        pad_len = s_len - MEAN_SIGNAL_LENGTH
        pad_len //= 2
        signal = signal[pad_len:pad_len + MEAN_SIGNAL_LENGTH]
    mel_coefficients = mfcc(signal, fs, num_cepstral=mfcc_len)
    if flatten:
        # Flatten the data
        mel_coefficients = np.ravel(mel_coefficients)
    return mel_coefficients