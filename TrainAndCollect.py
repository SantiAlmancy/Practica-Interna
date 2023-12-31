from Collections import Collections
from Preprocess import Preprocess
from Train import Train
from Detection import Detection
import numpy as np


#Datos
actions = np.array(['gracias','hola', 'porfavor', 'comer', 'pensar', 'amar'])
no_sequences = 30
sequence_length = 30
data_Path = 'MP_Data'
 
#Generar colecciones de datos
collections = Collections(actions, no_sequences, sequence_length, 1, data_Path)
collections.CreateFolders()
collections.Collect()


#Preprocesar datos
preprocess = Preprocess(actions, sequence_length, data_Path)
preprocess.PreProcessData()
print(preprocess.y_test.shape)

#Entrenar la red neuronal
train = Train(actions, preprocess.X_train, preprocess.y_train)
train.Training()

#Detección de Señas
detection = Detection(actions)
detection.recognition(preprocess.X_test)

