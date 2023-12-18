import cv2
import numpy as np
from Landmarks import Landmarks

class Camera:   

    def __init__(self):
        self.landmark = Landmarks()
        self.cap = None

    def AccessCamera(self):
        # Acceso a la webcam
        self.cap = cv2.VideoCapture(0)
        #min_detection_confidence = realizar una detección inicial para empezar el tracking
        #min_tracking_confidence = la confiabilidad del tracking
        with self.landmark.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # Mientras el webcam esté abierto
            while self.cap.isOpened():

                # Leer el feed, el frame (El while es rápido)
                ret, frame = self.cap.read()
                # Invertir horizontalmente la imagen
                frame = cv2.flip(frame, 1)

                # Make detections
                image, results = self.landmark.mediapipe_detection(frame, holistic)
                print(results)

                # Draw landmarks
                self.landmark.draw_styled_landmarks(image, results)

                # Mostrar en el screen
                #Image porque debe renderizar la nueva imagen, no el frame
                cv2.imshow('OpenCV Feed', image)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            self.cap.release()
            cv2.destroyAllWindows()

    #Extract keypoint values
    #pose = [] #Guarda todos los landmarks
    #for res in results.pose_landmarks.landmark:  #Para cada landmark en results (x, y, z, visibility)
    #flatten = para tener todos los landmarks en un array gigante, no varios arrays dentro de una mas grande (cada array para cada x, y, z, visibility)
    #if para controlar error en caso de que no se detecte ningun landmark para alguno de estos, se debe multiplicar la cantidad de landmarks por el tamaño de cada array que los guarda (se llena de ceros el array)

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4) #33 landmarks, 4 valores x landmark
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3) #468 landmarks, 3 valores x landmark
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3) #21 landmarks, 3 valores x landmark
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3) #21 landmarks, 3 valores x landmark
        return np.concatenate([pose, face, lh, rh]) #Concatenar todos los arrays