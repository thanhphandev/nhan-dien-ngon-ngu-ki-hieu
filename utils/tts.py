import threading
import time
import tempfile
import os

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    from gtts import gTTS
    from playsound import playsound
except Exception:
    gTTS = None
    playsound = None


class TextToSpeech:
    """TTS wrapper with two backends: 'pyttsx3' (offline) and 'gtts' (online, better Vietnamese).
    Includes debounce to avoid speaking every frame.
    """

    def __init__(self, engine: str = 'pyttsx3', lang: str = 'vi', rate: int | None = None,
                 volume: float | None = None, voice: str | None = None):
        self.engine_name = engine.lower()
        self.lang = lang

        # Backend initializations
        self._engine = None
        if self.engine_name == 'pyttsx3':
            if pyttsx3 is None:
                raise RuntimeError("pyttsx3 is not installed. Please install it or choose 'gtts'.")
            self._engine = pyttsx3.init()
            if rate is not None:
                self._engine.setProperty('rate', rate)
            if volume is not None:
                self._engine.setProperty('volume', volume)
            if voice is not None:
                for v in self._engine.getProperty('voices'):
                    if voice.lower() in (v.id or '').lower():
                        self._engine.setProperty('voice', v.id)
                        break
        elif self.engine_name == 'gtts':
            if gTTS is None or playsound is None:
                raise RuntimeError("gTTS/playsound not installed. Please install requirements or choose 'pyttsx3'.")
        else:
            raise ValueError("engine must be 'pyttsx3' or 'gtts'")

        self._lock = threading.Lock()
        self._busy = False
        self._last_text = None
        self._last_time = 0.0

    # --- Backend implementations ---
    def _speak_blocking_pyttsx3(self, text: str):
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        finally:
            with self._lock:
                self._busy = False
                self._last_text = text
                self._last_time = time.time()

    def _speak_blocking_gtts(self, text: str):
        tmp_path = None
        try:
            tts = gTTS(text=text, lang=self.lang)
            fd, tmp_path = tempfile.mkstemp(suffix='.mp3', prefix='tts_')
            os.close(fd)
            tts.save(tmp_path)
            playsound(tmp_path)
        except Exception:
            pass
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            with self._lock:
                self._busy = False
                self._last_text = text
                self._last_time = time.time()

    def speak_if_allowed(self, text: str, min_interval: float = 2.0):
        """Speak only if enough time has passed since the last utterance and text changed.
        Run speech in a background thread to avoid blocking the UI.
        """
        now = time.time()
        with self._lock:
            if self._busy:
                return
            if self._last_text == text and (now - self._last_time) < (min_interval * 3):
                return
            if (now - self._last_time) < min_interval:
                return
            self._busy = True

        if self.engine_name == 'pyttsx3':
            t = threading.Thread(target=self._speak_blocking_pyttsx3, args=(text,), daemon=True)
        else:
            t = threading.Thread(target=self._speak_blocking_gtts, args=(text,), daemon=True)
        t.start()

    def stop(self):
        try:
            if self.engine_name == 'pyttsx3' and self._engine is not None:
                self._engine.stop()
        except Exception:
            pass
