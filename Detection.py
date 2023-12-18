from Camera import Camera
from Landmarks import Landmarks
from TextToVoice import TexToSpeech
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import imutils
import threading


class Detection:

    def __init__(self, actions):
        self.Colors = [(245,117,16), (117,245,16), (16,117,245)]
        self.Actions = actions
        self.landmark = Landmarks()
        self.camera = Camera()
        self.TexToSpeech = TexToSpeech('HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-MX_SABINA_11.0', 200, 1.0)
        self.running = False
        self.scan = False
        self.LastText = ""
        self.thread_tts = None
        self.sound = True

    def prob_viz(self, res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
        return output_frame
    
    def recognition(self, res):
        model = load_model('model/mi_modelo.h5')
        res = model.predict(res)
        # Holistic model: para realizar la detección
        mp_holistic = mp.solutions.holistic
        # Drawing utilities: para dibujar lo que se detecta
        mp_drawing = mp.solutions.drawing_utils

        # 1. New detection variables
        sequence = []
        sentence = []
        predictions = []
        threshold = 0.5

        cap = cv2.VideoCapture(0)
        # Set mediapipe model 
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = self.landmark.mediapipe_detection(frame, holistic)
                print(results)
                
                # Draw landmarks
                if self.scan:
                    self.landmark.draw_styled_landmarks(image, results)
                
                # 2. Prediction logic
                keypoints = self.camera.extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
                
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(self.Actions[np.argmax(res)])
                    predictions.append(np.argmax(res))
                    
                    
                #3. Viz logic
                    if np.unique(predictions[-10:])[0]==np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 
                            
                            if len(sentence) > 0: 
                                if self.Actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(self.Actions[np.argmax(res)])
                            else:
                                sentence.append(self.Actions[np.argmax(res)])

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]

                    # Viz probabilities
                    image = self.prob_viz(res, self.Actions, image, self.Colors)
                    
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Show to screen
                cv2.imshow('OpenCV Feed', image)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

    def recognitionCanvas(self, res, canvas, textBox, app):
        self.running = True
        model = load_model('model/mi_modelo.h5')
        res = model.predict(res)
        # Holistic model: para realizar la detección
        mp_holistic = mp.solutions.holistic
        # Drawing utilities: para dibujar lo que se detecta
        mp_drawing = mp.solutions.drawing_utils

        # 1. New detection variables
        sequence = []
        sentence = []
        predictions = []
        threshold = 0.5

        cap = cv2.VideoCapture(0)
        # Set mediapipe model 
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened() and self.running:

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = self.landmark.mediapipe_detection(frame, holistic)
                print(results)
                
                # Draw landmarks
                if self.scan:
                    self.landmark.draw_styled_landmarks(image, results)
                
                # 2. Prediction logic
                keypoints = self.camera.extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
                
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(self.Actions[np.argmax(res)])
                    predictions.append(np.argmax(res))
                    
                    
                #3. Viz logic
                    if np.unique(predictions[-10:])[0]==np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 
                            
                            if len(sentence) > 0: 
                                if self.Actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(self.Actions[np.argmax(res)])
                            else:
                                sentence.append(self.Actions[np.argmax(res)])

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]

                    # Viz probabilities
                    #image = self.prob_viz(res, self.Actions, image, self.Colors)
                    if (not self.Actions[np.argmax(res)] == self.LastText):
                        self.LastText = self.Actions[np.argmax(res)]
                        app.itemconfig(textBox, text=self.Actions[np.argmax(res)])
                        if self.thread_tts is None or not self.thread_tts.is_alive() and self.sound:
                            self.thread_tts = threading.Thread(target=self.TexToSpeech.text_to_sound, args=(self.LastText,))
                            self.thread_tts.start()

                    
                #cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                #cv2.putText(image, ' '.join(sentence), (3,30), 
                            #cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Show to screen
                #cv2.imshow('OpenCV Feed', image)
                image = imutils.resize(image, width=1035)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                im = Image.fromarray(image)
                img = ImageTk.PhotoImage(image=im)

                canvas.configure(image=img)
                canvas.image = img


            
            cap.release()
            cv2.destroyAllWindows()

    def start_recognition(self, res, canvas):
        thread = threading.Thread(target=self.recognitionCanvas, args=(res, canvas))
        thread.start()

    def stopVideo(self):
        self.running = False

    def scanning(self):
        self.scan = not self.scan

    def voiceSound(self):
        self.sound = not self.sound

    def changeVoice(self):
        self.TexToSpeech.change_voice()