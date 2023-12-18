import pyttsx3

class TexToSpeech:
    engine: pyttsx3.Engine

    def __init__(self, voice, rate: int, volume: float):
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
        self.current_voice_index = 0

        if voice:
            self.engine.setProperty('voice', voice)
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


#if __name__ == '__main__':
#    tts = TexToSpeech('HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-MX_SABINA_11.0', 200, 1.0)
#    tts.list_available_voice()
#    tts.text_to_sound("Hola, me llamo Lucas")


#HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-MX_SABINA_11.0