from voice_auth.recognize import recognize
from sentiment_analysis.classify import sentiment_classify
import os

def authenticate():
    username = input("Enter authentication username: ")
    if not os.path.exists('./gmm_models/' + username + '.gmm'):
        print("User doesn't exist!")
        return False
    
    likelihood = recognize(username)
    
    if likelihood > -23:
        sentiment_result = sentiment_classify('./test.wav')
        if sentiment_result == 'Neutral' or sentiment_result == 'Happy':
            print(f'Authentication Successful for {username}!')
            return True
        else:
            print('User could be under influence!')
    else:
        print('Biometric Authentication Failed, please try again!')
    
    return False