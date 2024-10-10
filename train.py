from data_loader import load_data
from model import build_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

def train_model(data_dir, model_save_path):
    """
    Trains the DNN model.

    Parameters:
        data_dir (str): Directory containing training data.
        model_save_path (str): Path to save the trained model.
    """
    # Load data
    X, y = load_data(data_dir)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Build model
    input_shape = X_train.shape[1]
    model = build_model(input_shape)

    # Set up checkpointing
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss',
                                 save_best_only=True, verbose=1)

    # Train model
    model.fit(X_train, y_train, batch_size=32, epochs=50,
              validation_data=(X_val, y_val), callbacks=[checkpoint])

if __name__ == '__main__':
    data_dir = 'data/train'  # Replace with your training data directory
    model_save_path = 'saved_models/audio_separator.h5'
    train_model(data_dir, model_save_path)
