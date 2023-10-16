from feature_vector import get_feature_vector_from_mfcc
import numpy as np
import tensorflow as tf

LABELS = ["Neutral", "Angry", "Happy", "Sad"]

def sentiment_classify(filename):
    test_features = get_feature_vector_from_mfcc(filename, flatten=False)
    test_features = np.expand_dims(test_features, axis=2)

    model = tf.keras.models.load_model('../sentiment_models/sentiment_classifier.h5')
    prediction = np.argmax(model.predict(np.array([test_features])))
    predicted_label = LABELS[prediction]
    return predicted_label