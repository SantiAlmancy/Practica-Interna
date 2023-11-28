import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

class Train:
    def __init__(self, array, x_train, y_train) -> None:
        self.Actions = array
        self.X_train = x_train
        self.Y_train = y_train

    def Training(self):
        log_dir = os.path.join('Logs')
        tb_callback = TensorBoard(log_dir=log_dir)
        #Red neuronal
        # Modelo secuencial, fácil de crear y añadir modulos
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.Actions.shape[0], activation='softmax'))

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.fit(self.X_train, self.Y_train, epochs=2000, callbacks=[tb_callback])
        model.save('model/mi_modelo.h5')
        print(model.summary())