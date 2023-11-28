import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


class Preprocess:

    def __init__(self, actions, sequence_length, data_Path):
        self.Actions = actions
        self.Data_Path = data_Path
        self.Sequence_Length = sequence_length
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def PreProcessData (self):
        label_map = {label:num for num, label in enumerate(self.Actions)}
        #Sequences: Data
        #Labels: labels for data
        sequences, labels = [], []
        for action in self.Actions:
            for sequence in np.array(os.listdir(os.path.join(self.Data_Path, action))).astype(int):
                window = [] # ALL frames for a sequence
                for frame_num in range(self.Sequence_Length): # For each frame
                    # Add al frames to the window
                    res = np.load(os.path.join(self.Data_Path, action, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                labels.append(label_map[action])
        
        X = np.array(sequences)
        y = to_categorical(labels).astype(int)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.05)

