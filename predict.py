import numpy as np
from feature_extraction import extract_features
from tensorflow.keras.models import load_model
import librosa
import soundfile as sf

def separate_audio(audio_path, model_path):
    """
    Separates an audio file into vocals and instrumentals.

    Parameters:
        audio_path (str): Path to the input audio file.
        model_path (str): Path to the trained model file.
    """
    # Load the trained model
    model = load_model(model_path)

    # Extract features from the audio file
    features, y, sr = extract_features(audio_path)
    predictions = model.predict(features).flatten()

    # Create masks based on predictions
    mask = predictions > 0.5  # Binary mask
    mask = mask.astype(np.float32)

    # Perform STFT
    hop_length = 512
    n_fft = 2048
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(stft)

    # Apply masks
    vocal_magnitude = magnitude * mask[:, np.newaxis]
    instrumental_magnitude = magnitude * (1 - mask[:, np.newaxis])

    # Reconstruct signals
    vocal_stft = vocal_magnitude * phase
    instrumental_stft = instrumental_magnitude * phase

    vocal_signal = librosa.istft(vocal_stft, hop_length=hop_length)
    instrumental_signal = librosa.istft(instrumental_stft, hop_length=hop_length)

    # Save output files
    sf.write('vocals.wav', vocal_signal, sr)
    sf.write('instrumentals.wav', instrumental_signal, sr)

if __name__ == '__main__':
    audio_path = 'data/test/audio_file.wav'  # Replace with your test audio file
    model_path = 'saved_models/audio_separator.h5'
    separate_audio(audio_path, model_path)
