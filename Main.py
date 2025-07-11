import librosa
import numpy as np
from keras import models
import math
import json
model = models.load_model('CNN.h5')

def load_genre(json_path):

    with open(json_path, 'r') as file:
        data = json.load(file)
    return data['genres']

def predict_genre_from_file(model, file_path, genre_mapping, sample_rate=22050, num_mfcc=40, n_fft=2048, hop_length=512, num_segments=100):

    # Load audio file
    signal, sr = librosa.load(file_path, sr=sample_rate)

    # Ensure the audio file is long enough
    track_duration = 30  # Track duration in seconds (as expected by the model)
    samples_per_track = sample_rate * track_duration
    assert len(signal) >= samples_per_track, "The audio file is shorter than expected by the model!"

    samples_per_segment = int(samples_per_track / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    predictions = []

    # Process each segment of the audio file
    for d in range(num_segments):
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T

        # Make a prediction only if we have the expected number of MFCC vectors
        if len(mfcc) == num_mfcc_vectors_per_segment:
            mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions (if your model expects it)
            prediction = model.predict(mfcc)
            predicted_index = np.argmax(prediction, axis=1)
            predictions.append(predicted_index)

    # Assuming we use the mode of the predictions as the final class label
    from scipy.stats import mode
    final_prediction = mode(predictions)[0]
    final_prediction = final_prediction[0] if len(final_prediction) > 0 else None

    if final_prediction is not None:
        # Translate index to genre name
        genre = genre_mapping[int(final_prediction)]
    else:
        genre = "Prediction was not conclusive."

    return genre

# Example Usage
if __name__ == "__main__":
    genre_mapping = load_genre('DataMFCC.json')
    genre = predict_genre_from_file(model, 'rock.00079.wav', genre_mapping)
    print(f"Predicted Genre: {genre}")
