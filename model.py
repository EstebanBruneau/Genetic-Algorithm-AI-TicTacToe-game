import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

def create_model():
    model = Sequential([
        Flatten(input_shape=(9,)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(9, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model
