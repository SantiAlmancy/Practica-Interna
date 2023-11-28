import cv2
import numpy as np
import os
from Landmarks import Landmarks
from Camera import Camera

class Collections:

    def __init__(self, array, no_sequences, sequence_length, start_folder, data_Path):
        self.Array = array
        self.No_Sequences = no_sequences
        self.Sequence_Length = sequence_length
        self.Start_Folder = start_folder
        self.Data_Path = data_Path
        self.Landmark = Landmarks()
        self.Camera = Camera()

    def CreateFolders(self):
        # Path for exported data, numpy arrays
        DATA_PATH = self.Data_Path

        # Actions that we try to detect
        actions = self.Array

        # Thirty videos worth of data
        no_sequences = self.No_Sequences

        # Videos are going to be 30 frames in length
        sequence_length = self.Sequence_Length

        # Folder start
        start_folder = self.Start_Folder

        # Crear un folder por cada acción
        for action in actions:
            action_path = os.path.join(DATA_PATH, action)
            
            # Verificar si la carpeta 'MP_Data' existe
            if not os.path.exists(action_path):
                try:
                    os.makedirs(action_path)
                except Exception as e:
                    print(f"No se pudo crear la carpeta {action_path}. Error: {e}")
                    continue

            # Obtener el valor máximo en las carpetas existentes y usarlo como base
            dirmax = np.max([int(folder) for folder in os.listdir(action_path)]) if os.listdir(action_path) else 0

            # Crear carpetas para cada secuencia
            for sequence in range(1, no_sequences + 1):
                try:
                    os.makedirs(os.path.join(action_path, str(dirmax + sequence)))
                except Exception as e:
                    print(f"No se pudo crear la carpeta en {action_path}. Error: {e}")
                    continue

    def Collect(self):
        cap = cv2.VideoCapture(0)
        # Set mediapipe model 
        with self.Landmark.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
            # NEW LOOP
            # Loop through actions
            for action in self.Array:
                # Loop through sequences aka videos
                for sequence in range(self.Start_Folder, self.Start_Folder + self.No_Sequences):
                    # Loop through video length aka sequence length
                    for frame_num in range(self.Sequence_Length):

                        # Read feed
                        ret, frame = cap.read()

                        # Make detections
                        image, results = self.Landmark.mediapipe_detection(frame, holistic)

                        # Draw landmarks
                        self.Landmark.draw_styled_landmarks(image, results)
                        
                        # NEW Apply wait logic
                        if frame_num == 0: 
                            cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(500)
                        else: 
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                        
                        # NEW Export keypoints
                        keypoints = self.Camera.extract_keypoints(results)
                        npy_path = os.path.join(self.Data_Path, action, str(sequence), str(frame_num))
                        np.save(npy_path, keypoints)

                        # Break gracefully
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
                            
            cap.release()
            cv2.destroyAllWindows()