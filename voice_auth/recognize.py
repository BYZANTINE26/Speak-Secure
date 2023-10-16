import os
import pickle
import wave
import pyaudio
from scipy.io.wavfile import read
import numpy as np
from audio_processing import extract_features

def recognize(target_username):
    # target_username = input("Enter authentication username: ")
    # if not os.path.exists('./gmm_models/' + target_username + '.gmm'):
    #     print("User doesn't exist!")
    #     return
    # else:
    target_model = pickle.load(open('../gmm_models/' + target_username + '.gmm', 'rb'))

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3
    FILENAME = "../test.wav"

    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # saving wav file
    waveFile = wave.open(FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

# Find the GMM model for the target username
    # # target_model = None
    # # for fname in gmm_files:
    # #     speaker = fname.split("/")[-1].split(".gmm")[0]
    # #     if speaker == target_username:
    # #         target_model = pickle.load(open(fname, 'rb'))
    # #         break

    # # if target_model is None:
    # #     print("User not found in the database!")
    # #     return

    # Read test file
    sr, audio = read(FILENAME)

    # Extract MFCC features
    vector = extract_features(audio, sr)
    log_likelihood = target_model.score(vector)  # Calculate the log-likelihood for the target model
    # print(log_likelihood)
    return log_likelihood

    if log_likelihood > some_threshold:  # Set an appropriate threshold for recognition
        print(f"Recognized as {target_username}")
    else:
        print("Not Recognized! Try again...")

# Checking all models
    # modelpath = "../gmm_models/"
    # gmm_files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]
    # models = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    # speakers = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]

    # if len(models) == 0:
    #     print("No Users in the Database!")
    #     return
    
    # #read test file
    # sr,audio = read(FILENAME)

    # # extract mfcc features
    # vector = extract_features(audio,sr)
    # log_likelihood = np.zeros(len(models))

    # #checking with each model one by one
    # for i in range(len(models)):
    #     gmm = models[i]         
    #     scores = np.array(gmm.score(vector))
    #     log_likelihood[i] = scores.sum()

    # print(log_likelihood)

    # pred = np.argmax(log_likelihood)
    # identity = speakers[pred]
    # return identity, pred
    # print(identity)
    # print(speakers)

    # # if voice not recognized than terminate the process
    # if identity == 'unknown':
    #     print("Not Recognized! Try again...")
    #     return
    # else:
    #     print( "Recognized as - ", identity)
    #     return identity
