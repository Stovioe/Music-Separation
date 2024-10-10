import numpy as np
import librosa

def extract_features(audio_path, n_mfcc=13, hop_length=512, n_fft=2048):
    """
    Extracts acoustic features from an audio file.

    Parameters:
        audio_path (str): Path to the audio file.
        n_mfcc (int): Number of MFCCs to return.
        hop_length (int): Number of samples between successive frames.
        n_fft (int): Length of the FFT window.

    Returns:
        features (np.ndarray): Combined feature array.
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate.
    """
    y, sr = librosa.load(audio_path, sr=None)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                 hop_length=hop_length, n_fft=n_fft).T

    # Extract Spectral Features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr,
                                                          hop_length=hop_length, n_fft=n_fft).T
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr,
                                                            hop_length=hop_length, n_fft=n_fft).T
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr,
                                                          hop_length=hop_length, n_fft=n_fft).T
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr,
                                                        hop_length=hop_length, n_fft=n_fft).T

    # Extract Pitch
    pitches, _ = librosa.piptrack(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
    pitch_mean = np.mean(pitches, axis=0).reshape(-1, 1)

    # Stack all features
    features = np.hstack((mfccs, spectral_centroid, spectral_bandwidth,
                          spectral_contrast, spectral_rolloff, pitch_mean))

    return features, y, sr
