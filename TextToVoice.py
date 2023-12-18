import pyttsx3

class TexToSpeech:
    engine: pyttsx3.Engine

    def __init__(self, rate: int, volume: float):
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
        self.current_voice_index = 0

        if self.voices:
            self.engine.setProperty('voice', self.voices[0].id)
        else:
            print("No se encontraron voces disponibles.")

        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)

    def list_available_voices(self):
        for i, voice in enumerate(self.voices):
            print(f'{i + 1} {voice.name} {voice.age}: ({voice.gender}) [{voice.id}]')

    def change_voice(self):
        # Cambia a la siguiente voz disponible
        self.current_voice_index = (self.current_voice_index + 1) % len(self.voices)
        self.engine.setProperty('voice', self.voices[self.current_voice_index].id)


    def text_to_sound(self, text: str, save: bool = False, file_name = 'output.mp3'):
        self.engine.say(text)
        print('Estoy hablando ...')

        if save:
            self.engine.save_to_file(text, file_name)

        self.engine.runAndWait()