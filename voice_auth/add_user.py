import os
import pyaudio
import time
from IPython.display import clear_output
import wave
import numpy as np
from scipy.io.wavfile import read
from audio_processing import extract_features
from sklearn.mixture import GaussianMixture as GMM
import pickle

def add_user():
    name = input("Enter your username: ")
    if os.path.exists('./voice_database/' + name):
        print("User already exists \n Try again with another username")
        return
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3 # TODO: 5

    source = "../voice_database/" + name
    os.mkdir(source)
    for i in range(3):
        audio = pyaudio.PyAudio()

        if i == 0:
            j = 3
            while j>=0:
                time.sleep(1.0)
                print("Speak your name in {} seconds".format(j))
                clear_output(wait = True)
                j -= 1
        
        elif i == 1:
            print("Speak your name one more time")
            time.sleep(0.5)
        
        else:
            print("Speak your name one last time")
            time.sleep(0.5)
        
        # start Recording
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        print("recording...")
        frames = []

        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # saving wav file of speaker
        waveFile = wave.open(source + '/' + str((i+1)) + '.wav', 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        print("Done")

    dest =  "../gmm_models/"
    count = 1

    for path in os.listdir(source):
        path = os.path.join(source, path)

        features = np.array([])

        # reading audio files of speaker
        (sr, audio) = read(path)

        # extract 40 dimensional MFCC & delta MFCC features
        vector = extract_features(audio,sr)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

        # when features of 3 files of speaker are concatenated, then do model training
        if count == 3:
            gmm = GMM(n_components = 16, max_iter = 200, covariance_type='diag', n_init = 3)
            gmm.fit(features)

            with open(dest + name + '.gmm', 'wb') as file:
                pickle.dump(gmm, file)
            print(name + ' added successfully')

            features = np.asarray(())
            count = 0
        count = count + 1

