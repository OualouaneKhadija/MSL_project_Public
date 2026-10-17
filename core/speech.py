from pathlib import Path
import tempfile

import pygame
from gtts import gTTS
from googletrans import Translator


AUDIO_MAP = {
    "ka2ib":         "audio/ka2ib.mp3",
    "l3afw":         "audio/l3afw.mp3",
    "labas":         "audio/labas.mp3",
    "lhob":          "audio/lhob.mp3",
    "lmra":          "audio/lmra.mp3",
    "mghyar":        "audio/mghyar.mp3",
    "mounafiq":      "audio/mounafiq.mp3",
    "chokran":       "audio/chokran.mp3",
    "deki":          "audio/deki.mp3",
    "deri_sghir":    "audio/deri_sghir.mp3",
    "farhan":        "audio/farhan.mp3",
    "intiwai":       "audio/intiwai.mp3",
    "jari":          "audio/jari.mp3",
    "3yan":          "audio/3yan.mp3",
    "7chman":        "audio/7chman.mp3",
    "salam_alaykom": "audio/salam_alaykom.mp3",
}

LANGUAGE_CODES = {
    "darija":   None,
    "français": "fr",
    "anglais":  "en",
    "espagnol": "es",
    "chinois":  "zh-CN",
}


class SpeechService:
    def __init__(self, language="darija"):
        self.language = language
        self.translator = Translator()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="msl_audio_"))
        pygame.mixer.init()

    def set_language(self, language):
        if language in LANGUAGE_CODES:
            self.language = language

    def speak(self, text, label=None):
        if not text:
            return

        if self.language == "darija":
            if label and label in AUDIO_MAP:
                path = AUDIO_MAP[label]
                if Path(path).exists():
                    pygame.mixer.music.stop()
                    pygame.mixer.music.load(path)
                    pygame.mixer.music.play()
                    return

        lang_code = LANGUAGE_CODES.get(self.language, "fr")

        try:
            translated = self.translator.translate(text, src="ar", dest=lang_code).text
        except Exception:
            translated = text

        tmp = self.temp_dir / "speech.mp3"
        gTTS(text=translated, lang=lang_code).save(str(tmp))

        pygame.mixer.music.stop()
        pygame.mixer.music.load(str(tmp))
        pygame.mixer.music.play()

    def close(self):
        pygame.mixer.quit()
