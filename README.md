# Music-Separation

Model Architecture:
We employed a multi-layer DNN where the input layer took in various acoustic features extracted from the audio signal, such as MFCCs (Mel-Frequency Cepstral Coefficients), spectral features, and pitch. These features capture the timbral and temporal characteristics of the sound, making it easier to distinguish between speech and music.
The network consisted of multiple hidden layers, using non-linear activation functions (ReLU: rectified linear unit) to capture the complex relationships between the input features and the desired output, i.e., separating the audio into its component parts.
The output layer produced a prediction for each time frame, classifying the sound as either vocal or instrumental. For this task, we used sigmoid activation to ensure the output was binary.

Training Process:
We trained the model on a large dataset of labeled audio files with separated vocals and instrumentals. During training, we used backpropagation and the Adam optimizer to minimize the binary cross-entropy loss function.
To ensure robustness, we applied techniques like batch normalization, dropout, and data augmentation (adding noise, pitch shifting) to prevent overfitting and improve generalization.

How the Project Works:
The process begins by feeding an audio clip through a feature extraction pipeline that converts the raw waveform into a set of acoustic features. These features are then passed through the DNN, which learns the latent patterns that differentiate speech from music.
The trained model predicts frame-by-frame probabilities for vocals or instrumentals, and the output is reconstructed to produce two separate audio streams.
Post-processing, such as time-frequency masking, is used to refine the separation by enhancing the vocal/instrumental distinction in overlapping frequency bands.
