import numpy as np
import glob
import os
from feature_extraction import extract_features
import librosa

def add_noise(y, noise_factor=0.005):
    """
    Adds white noise to an audio signal.

    Parameters:
        y (np.ndarray): Audio time series.
        noise_factor (float): Amount of noise to add.

    Returns:
        np.ndarray: Noisy audio signal.
    """
    noise = np.random.randn(len(y))
    augmented_data = y + noise_factor * noise
    return augmented_data.astype(type(y[0]))

def shift_pitch(y, sr, pitch_factor):
    """
    Shifts the pitch of an audio signal.

    Parameters:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate.
        pitch_factor (float): Number of half-steps to shift.

    Returns:
        np.ndarray: Pitch-shifted audio signal.
    """
    return librosa.effects.pitch_shift(y, sr, n_steps=pitch_factor)

def load_data(data_dir, augmentation=True):
    """
    Loads data from a directory and applies augmentation.

    Parameters:
        data_dir (str): Directory containing audio files.
        augmentation (bool): Whether to apply data augmentation.

    Returns:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
    """
    X = []
    y = []
    audio_files = glob.glob(os.path.join(data_dir, '*.wav'))

    for file in audio_files:
        # Determine label from filename
        label = 1 if 'vocals' in file else 0

        # Extract features
        features, audio, sr = extract_features(file)
        X.append(features)
        y.append(np.full(features.shape[0], label))

        if augmentation:
            # Augment with noise
            noisy_audio = add_noise(audio)
            features_noisy, _, _ = extract_features_from_audio(noisy_audio, sr)
            X.append(features_noisy)
            y.append(np.full(features_noisy.shape[0], label))

            # Augment with pitch shifting
            for pitch_factor in [-2, -1, 1, 2]:
                shifted_audio = shift_pitch(audio, sr, pitch_factor)
                features_shifted, _, _ = extract_features_from_audio(shifted_audio, sr)
                X.append(features_shifted)
                y.append(np.full(features_shifted.shape[0], label))

    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

def extract_features_from_audio(y, sr, n_mfcc=13, hop_length=512, n_fft=2048):
    """
    Extracts features from raw audio data.

    Parameters:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate.

    Returns:
        features (np.ndarray): Feature array.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                 hop_length=hop_length, n_fft=n_fft).T

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr,
                                                          hop_length=hop_length, n_fft=n_fft).T
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr,
                                                            hop_length=hop_length, n_fft=n_fft).T
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr,
                                                          hop_length=hop_length, n_fft=n_fft).T
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr,
                                                        hop_length=hop_length, n_fft=n_fft).T

    pitches, _ = librosa.piptrack(y=y, sr=sr,
                                  hop_length=hop_length, n_fft=n_fft)
    pitch_mean = np.mean(pitches, axis=0).reshape(-1, 1)

    features = np.hstack((mfccs, spectral_centroid, spectral_bandwidth,
                          spectral_contrast, spectral_rolloff, pitch_mean))

    return features, y, sr
