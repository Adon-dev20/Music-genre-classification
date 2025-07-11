import librosa
import json
import os

audios = "genres_original"
dataset = "MFCC.json"
sr = 22050
duration = 30
expected_duration = sr * duration


def check_audio_length(signal, expected_duration):
    if len(signal) >= expected_duration:
        return signal[:expected_duration]
    return None


def extract_mfcc(dataset_path, json_path, n_mfcc=40, n_fft=2048, hop_length=512, num_segments=100):
    data = {
        "genres": [],
        "labels": [],
        "mfccs": []
    }
    segment = int(expected_duration / num_segments)

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath != dataset_path:
            genre_name = os.path.basename(dirpath)
            data["genres"].append(genre_name)

            for j in filenames:
                file_path = os.path.join(dirpath, j)
                signal, sample_rate = librosa.load(file_path, sr=sr)

                signal = check_audio_length(signal, expected_duration)
                if signal is None:
                    print(f"Ignoring {file_path}: Audio is shorter than 30 seconds.")
                    continue

                for d in range(num_segments):
                    begin = segment * d
                    end = begin + segment

                    mfcc = librosa.feature.mfcc(y=signal[begin:end], sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    data["labels"].append(i - 1)
                    data["mfccs"].append(mfcc.tolist())

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    extract_mfcc(audios, dataset, num_segments=100)
