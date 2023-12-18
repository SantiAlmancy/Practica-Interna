from pathlib import Path
from tkinter import Tk, Canvas, Button, PhotoImage, Label
from Collections import Collections
from Preprocess import Preprocess
from Detection import Detection
import numpy as np
import threading



OUTPUT_PATH = Path(__file__).parent

# Combina el directorio actual con la ruta relativa a 'assets'
ASSETS_PATH = OUTPUT_PATH / "assets" / "frame0"


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("1593x1010")
window.configure(bg = "#282634")

canvas = Canvas(
    window,
    bg = "#282634",
    height = 1010,
    width = 1593,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    0.0,
    0.0,
    96.0,
    1010.0,
    fill="#F0F0F0",
    outline="")



result  = canvas.create_text(
    193.0,
    938.0,
    anchor="nw",
    text="",
    fill="#FFFFFF",
    font=("Aldrich Regular", 18 * -1)
)

canvas.create_text(
    119.0,
    21.0,
    anchor="nw",
    text="SIGN LANGUAGE DETECTION",
    fill="#F80000",
    font=("Aldrich Regular", 18 * -1)
)

canvas.create_rectangle(
    1169.0,
    24.0,
    1571.0,
    969.0,
    fill="#F0F0F0",
    outline="")

canvas.create_text(
    1189.0,
    52.0,
    anchor="nw",
    text="1 Gracias\n2 Hola\n3 Por favor\n4 Comer\n5 Pensar\n6 Amar",
    fill="#2C2C2C",
    font=("Aldrich Regular", 18 * -1)
)

image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    146.0,
    946.0,
    image=image_image_1
)

canvas.create_rectangle(
    95.0,
    0.0,
    96.00000002400316,
    1029.0,
    fill="#FFFFFF",
    outline="")

canvas.create_rectangle(
    95.0,
    66.0,
    1156.0,
    67.0,
    fill="#FBFBFB",
    outline="")


#Video
video_tq = Label(window, bg="black")
video_tq.place(x=114, y=95)

#Datos
actions = np.array(['gracias','hola', 'porfavor', 'comer', 'pensar', 'amar'])

no_sequences = 30
sequence_length = 30
data_Path = 'MP_Data'

collections = Collections(actions, no_sequences, sequence_length, 1, data_Path)

preprocess = Preprocess(actions, sequence_length, data_Path)
preprocess.PreProcessData()

detection_instance = Detection(actions)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_1 clicked"),
    relief="flat"
)
button_1.place(
    x=14.0,
    y=63.0,
    width=68.0,
    height=66.0
)

def on_button_2_click():
    detection_instance.changeVoice()

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=on_button_2_click,
    relief="flat"
)
button_2.place(
    x=14.0,
    y=343.0,
    width=68.0,
    height=71.0
)

def on_button_5_click():
    detection_instance.voiceSound()

button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=on_button_5_click,
    relief="flat"
)
button_4.place(
    x=15.0,
    y=53.0,
    width=68.0,
    height=87.0
)

running = False

def on_button_5_click():
    global running, detection_instance
    
    if not running:
        thread = threading.Thread(target=detection_instance.recognitionCanvas, args=(preprocess.X_test, video_tq, result, canvas))
        thread.start()
        running = True
    else:
        # Detener la cámara si ya está en funcionamiento
        if detection_instance:
            detection_instance.stopVideo()
            running = False


button_image_5 = PhotoImage(
    file=relative_to_assets("button_5.png"))
button_5 = Button(
    image=button_image_5,
    borderwidth=0,
    highlightthickness=0,
    command=on_button_5_click,
    relief="flat"
    
)
button_5.place(
    x=19.0,
    y=473.0,
    width=61.0,
    height=65.0
)

def on_button_6_click():
    detection_instance.scanning()

button_image_6 = PhotoImage(
    file=relative_to_assets("button_6.png"))
button_6 = Button(
    image=button_image_6,
    borderwidth=0,
    highlightthickness=0,
    command=on_button_6_click,
    relief="flat"
)
button_6.place(
    x=19.0,
    y=608.0,
    width=61.0,
    height=82.0
)



window.resizable(False, False)
window.mainloop()
