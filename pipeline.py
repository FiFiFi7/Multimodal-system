class AudioPipeline:
    def __init__(self):
        self.noise = FullSubNet()
        self.asr = WhisperModel("distil-large-v3")
        self.translator = M2M100("418M")
        self.emotion = SER()
        self.tts = ElevenLabs()

    def run(self, audio, mode, target_lang=None, target_emotion=None):
        clean = self.noise(audio)
        text = self.asr(clean)

        if mode == "accent":
            return self.tts.speak(text, voice="neutral")

        if mode == "translate":
            tgt = self.translator(text, tgt_lang=target_lang)
            return self.tts.speak(tgt)

        if mode == "emotion_convert":
            return self.tts.speak(text, emotion=target_emotion)

        if mode == "full":
            emo = self.emotion(clean)
            translated = self.translator(text, target_lang)
            return self.tts.speak(translated, emotion=target_emotion)
