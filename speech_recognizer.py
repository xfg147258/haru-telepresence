"""Realtime speech recognition for local microphone and remote audio capture."""

import time
from collections import deque
from threading import Thread

import numpy as np
import pyaudio
import webrtcvad

from audio_manager import EnhancedPulseAudioManager
from constants import (
    AUDIO_CHANNELS,
    AUDIO_CHUNK,
    AUDIO_RATE,
    MAX_AUDIO_ERRORS,
    MIN_SPEECH_LENGTH,
    PROCESS_INTERVAL,
    SILENCE_THRESHOLD,
    SILENCE_TIMEOUT,
    SPEECH_THRESHOLD,
)

_VAD_MODE = 2
_AUDIO_LEVEL_HISTORY_LEN = 50
_AUDIO_LEVEL_REPORT_INTERVAL_S = 10.0
_SHORT_TEXT_LEN = 8


class RealtimeStreamingSpeechRecognizer:
    """Realtime recognizer using WebRTC VAD + Whisper.

    Captures audio from the default input device, segments speech using
    voice-activity detection, and runs Whisper transcription on a background
    thread. Recognized text is forwarded to a user-supplied callback.
    """

    def __init__(self, callback, model_type: str = 'small',
                 language: str = 'en', engine: str = 'whisper') -> None:
        # Audio config.
        self.CHUNK = AUDIO_CHUNK
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = AUDIO_CHANNELS
        self.RATE = AUDIO_RATE

        # Recognition timing.
        self.MIN_SPEECH_LENGTH = MIN_SPEECH_LENGTH
        self.SILENCE_TIMEOUT = SILENCE_TIMEOUT
        self.PROCESS_INTERVAL = PROCESS_INTERVAL

        # VAD / speech state.
        self.vad: webrtcvad.Vad | None = webrtcvad.Vad(_VAD_MODE)
        self.speech_buffer: deque = deque()
        self.last_voice_time = time.time()
        self.is_speaking = False
        self.speech_start_time: float | None = None
        self.last_process_time = 0.0

        # Control flags.
        self.running = False
        self.processing = False
        self.callback = callback

        # Deduplication.
        self.last_sent_text = ''
        self.sent_count = 0

        # Recognition engine.
        self.engine = engine
        self.model_type = model_type
        self.language = language

        if engine == 'whisper':
            import whisper
            self.model = whisper.load_model(model_type)
            print(f'Whisper {model_type} model loaded.')

    # ---------------------------------------------------------------------
    # Deduplication / buffer management
    # ---------------------------------------------------------------------

    def should_send_text(self, text: str) -> bool:
        """Returns True if `text` is non-empty and substantively new."""
        if not text or len(text.strip()) < 2:
            return False
        clean = text.strip().lower()
        last = self.last_sent_text.strip().lower()
        if clean == last:
            return False
        if len(clean) < _SHORT_TEXT_LEN and clean in last:
            return False
        return True

    def clear_speech_buffer(self) -> None:
        self.speech_buffer.clear()
        self.is_speaking = False
        self.speech_start_time = None
        self.last_process_time = time.time()

    def reset_for_next_speech(self) -> None:
        self.is_speaking = False
        self.speech_start_time = None
        self.last_process_time = time.time()
        self.speech_buffer.clear()

    def reset(self) -> None:
        """Fully resets recognition state (between sessions)."""
        self.last_sent_text = ''
        self.sent_count = 0
        self.processing = False
        self.clear_speech_buffer()

    def stop(self) -> None:
        self.running = False

    # ---------------------------------------------------------------------
    # Audio callback
    # ---------------------------------------------------------------------

    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback: appends speech chunks and triggers processing."""
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        now = time.time()

        is_speech = self.vad.is_speech(in_data, self.RATE)

        if is_speech:
            self._on_speech_chunk(audio_data, now)
        elif self.is_speaking:
            silence_duration = now - self.last_voice_time
            if silence_duration >= self.SILENCE_TIMEOUT:
                self.clear_speech_buffer()

        return (in_data, pyaudio.paContinue)

    def _on_speech_chunk(self, audio_data: np.ndarray, now: float) -> None:
        self.last_voice_time = now
        if not self.is_speaking:
            self.is_speaking = True
            self.speech_start_time = now

        self.speech_buffer.append(audio_data.copy())

        speech_duration = now - self.speech_start_time if self.speech_start_time else 0
        time_since_process = now - self.last_process_time
        if (speech_duration >= self.MIN_SPEECH_LENGTH
                and time_since_process >= self.PROCESS_INTERVAL
                and not self.processing):
            self.last_process_time = now
            self.process_current_speech()

    # ---------------------------------------------------------------------
    # Recognition
    # ---------------------------------------------------------------------

    def process_current_speech(self) -> None:
        if not self.speech_buffer or self.processing:
            return
        chunks = list(self.speech_buffer)
        self.speech_buffer.clear()
        if not chunks:
            return
        audio = np.concatenate(chunks)
        Thread(target=self._transcribe_and_send, args=(audio,), daemon=True).start()

    def _transcribe_and_send(self, audio_np: np.ndarray) -> None:
        if self.processing:
            return
        self.processing = True
        try:
            text = self._transcribe(audio_np)
            if text and self.should_send_text(text):
                print(f'[TTS] {text}')
                self.last_sent_text = text
                self.sent_count += 1
                if self.callback:
                    self.callback(text, self.language)
                self.reset_for_next_speech()
        except Exception as exc:  # noqa: BLE001
            print(f'Recognition error: {exc}')
        finally:
            self.processing = False

    def _transcribe(self, audio_np: np.ndarray) -> str:
        if self.engine != 'whisper':
            return ''
        audio_float = audio_np.astype(np.float32) / 32768.0
        result = self.model.transcribe(
            audio_float,
            language=self.language if self.language != 'auto' else None,
            task='transcribe',
            fp16=False,
            verbose=False,
            beam_size=1,
            best_of=1,
            temperature=0,
        )
        return result['text'].strip()

    # ---------------------------------------------------------------------
    # Stream lifecycle
    # ---------------------------------------------------------------------

    def start_realtime_recognition(self) -> None:
        self.running = True
        try:
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=self.audio_callback,
            )
            print('Realtime speech recognition started.')
            stream.start_stream()
            while stream.is_active() and self.running:
                time.sleep(0.1)
            stream.stop_stream()
            stream.close()
            pa.terminate()
        except Exception as exc:  # noqa: BLE001
            print(f'Speech recognition failed to start: {exc}')
            self.running = False


class RemoteAudioSpeechRecognizer(RealtimeStreamingSpeechRecognizer):
    """Realtime recognizer that listens on a system-monitor audio device.

    Used in teleconference mode to transcribe the remote participant's voice.
    Replaces VAD with an RMS-level threshold because monitor streams typically
    do not deliver clean 10/20/30-ms frames.
    """

    def __init__(self, callback, model_type: str = 'small',
                 language: str = 'en', engine: str = 'whisper') -> None:
        super().__init__(callback, model_type, language, engine)

        # Disable VAD; use level-based detection instead.
        self.vad = None
        self.speech_threshold = SPEECH_THRESHOLD
        self.silence_threshold = SILENCE_THRESHOLD

        self.audio_level_history: deque = deque(maxlen=_AUDIO_LEVEL_HISTORY_LEN)
        self.last_audio_report = time.time()
        self.audio_error_count = 0
        self.max_audio_errors = MAX_AUDIO_ERRORS

        self.audio_manager = EnhancedPulseAudioManager()
        self.audio_device_index: int | None = None

    # ---------------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------------

    def setup_remote_audio(self) -> bool:
        return self.audio_manager.setup_monitor_device()

    @staticmethod
    def calculate_audio_level(audio_data: np.ndarray) -> float:
        if len(audio_data) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)))

    # ---------------------------------------------------------------------
    # Audio callback (level-based)
    # ---------------------------------------------------------------------

    def audio_callback(self, in_data, frame_count, time_info, status):
        if status:
            self.audio_error_count += 1
            if self.audio_error_count > self.max_audio_errors:
                self.running = False
                return (in_data, pyaudio.paAbort)

        try:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            now = time.time()

            level = self.calculate_audio_level(audio_data)
            self.audio_level_history.append(level)
            self._maybe_report_level(now)

            if level > self.speech_threshold:
                self._on_speech_chunk(audio_data, now)
            elif self.is_speaking:
                silence_duration = now - self.last_voice_time
                if silence_duration >= self.SILENCE_TIMEOUT:
                    if self.speech_buffer and not self.processing:
                        self.process_current_speech()
                    else:
                        self.clear_speech_buffer()

            return (in_data, pyaudio.paContinue)

        except Exception as exc:  # noqa: BLE001
            print(f'Audio callback error: {exc}')
            self.audio_error_count += 1
            return (in_data, pyaudio.paContinue)

    def _maybe_report_level(self, now: float) -> None:
        if now - self.last_audio_report <= _AUDIO_LEVEL_REPORT_INTERVAL_S:
            return
        avg = np.mean(list(self.audio_level_history)) if self.audio_level_history else 0
        print(f'Remote audio level: {avg:.1f}')
        self.last_audio_report = now

    # ---------------------------------------------------------------------
    # Stream lifecycle
    # ---------------------------------------------------------------------

    def start_realtime_recognition(self) -> None:
        print('\n=== Starting remote audio recognition ===')
        if not self.setup_remote_audio():
            print('Remote audio setup failed; aborting recognition.')
            return

        self.running = True
        self.audio_device_index = self.audio_manager.monitor_device_index
        if self.audio_device_index is None:
            print('No valid audio device index.')
            return

        pa = pyaudio.PyAudio()
        try:
            device_info = pa.get_device_info_by_index(self.audio_device_index)
            print(f"Using remote audio device: {device_info.get('name', 'Unknown')}")

            stream = self._open_stream_with_fallbacks(pa, device_info)
            if stream is None:
                print('Could not open remote audio stream.')
                return

            print('Remote audio recognition started.')
            stream.start_stream()
            while stream.is_active() and self.running:
                time.sleep(0.1)
            stream.stop_stream()
            stream.close()
        except Exception as exc:  # noqa: BLE001
            print(f'Remote recognition failed: {exc}')
            self.running = False
        finally:
            pa.terminate()

    def _open_stream_with_fallbacks(self, pa: pyaudio.PyAudio, device_info: dict):
        """Tries a few sample-rate / chunk-size combinations until one works."""
        configs = [
            (16000, 1, self.CHUNK),
            (44100, 1, 1024),
            (48000, 1, 1024),
            (int(device_info['defaultSampleRate']), 1, 1024),
        ]
        for rate, channels, chunk in configs:
            try:
                self.RATE = rate
                self.CHUNK = chunk
                self.CHANNELS = min(channels, device_info['maxInputChannels'])
                stream = pa.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True,
                    input_device_index=self.audio_device_index,
                    frames_per_buffer=self.CHUNK,
                    stream_callback=self.audio_callback,
                )
                print(f'Stream opened at {rate} Hz, {self.CHANNELS} channel(s).')
                return stream
            except Exception:  # noqa: BLE001
                continue
        return None
