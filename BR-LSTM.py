import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.layers import LSTM, Dense, Dropout
from tensorflow.python.keras.models import Sequential


from sklearn.model_selection import train_test_split
LABELS = [
    "FALL",
    "FIGHT",
    "STAND",
    "WALK"
]

X_train_path =  "trainx.txt"

y_train_path =   "trainy.txt"

n_steps = 15 # 15 timesteps per series


def load_X(X_path):
    file = open(X_path, 'r')
    X_ = np.array(
        [elem for elem in [
            row.split(',') for row in file
        ]],
        dtype=np.float32
    )
    file.close()
    blocks = int(len(X_) / n_steps)

    X_ = np.array(np.split(X_, blocks))

    return X_


def load_y(y_path):
    file = open(y_path, 'r')
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    return y_


X_data = load_X(X_train_path)

y_data = load_y(y_train_path)

print(X_data.shape, y_data.shape)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0)


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dense(units=4, activation='softmax'))


model.summary()


from tensorflow.python.keras.callbacks import ModelCheckpoint
epochs=30
# best_filepath = 'my_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
best_filepath = '1.hdf5'
checkpoint = ModelCheckpoint(filepath=best_filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

model.compile(optimizer="adam", metrics=['accuracy'], loss="sparse_categorical_crossentropy")


model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))
model.save(best_filepath)
