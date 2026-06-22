"""PulseAudio / PyAudio device discovery for remote audio capture."""

import subprocess

import numpy as np
import pyaudio

from constants import AUDIO_CHANNELS, AUDIO_RATE

_COMMAND_TIMEOUT_S = 10
_PROBE_BUFFER_SIZE = 1024
_MONITOR_KEYWORDS = ('monitor', 'loopback', 'what-u-hear', 'pulse', 'default')


class EnhancedPulseAudioManager:
    """Discovers and selects an input device suitable for capturing remote audio.

    The "monitor" device captures the system playback output, allowing the
    speech recognizer to listen to remote participants in a video call.
    """

    def __init__(self) -> None:
        self.monitor_source: str | None = None
        self.monitor_device_index: int | None = None
        self.available_devices: list = []

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _run(cmd: list[str]) -> tuple[str, bool]:
        """Runs a shell command, returning (stdout, success)."""
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                check=True, timeout=_COMMAND_TIMEOUT_S,
            )
            return result.stdout.strip(), True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(f"Command failed: {' '.join(cmd)}: {exc}")
            return '', False

    # ---------------------------------------------------------------------
    # Discovery
    # ---------------------------------------------------------------------

    def list_all_audio_devices(self) -> None:
        """Prints a human-readable inventory of audio devices."""
        print('\n=== Audio device inventory ===')

        print('\nPulseAudio sinks:')
        stdout, ok = self._run(['pactl', 'list', 'sinks', 'short'])
        if ok and stdout:
            for line in stdout.split('\n'):
                if line.strip():
                    print(f'  {line}')

        print('\nPulseAudio sources:')
        stdout, ok = self._run(['pactl', 'list', 'sources', 'short'])
        if ok and stdout:
            for line in stdout.split('\n'):
                if line.strip():
                    print(f'  {line}')

        print('\nPyAudio input devices:')
        try:
            pa = pyaudio.PyAudio()
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    print(f"  [{i}] {info['name']} "
                          f"(channels={info['maxInputChannels']}, "
                          f"rate={info['defaultSampleRate']})")
            pa.terminate()
        except Exception as exc:  # noqa: BLE001
            print(f'  PyAudio enumeration failed: {exc}')

    def find_monitor_devices(self) -> list[dict]:
        """Returns candidate monitor/loopback devices from PulseAudio + PyAudio."""
        candidates: list[dict] = []

        stdout, ok = self._run(['pactl', 'list', 'sources', 'short'])
        if ok:
            for line in stdout.split('\n'):
                if line.strip() and 'monitor' in line.lower():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        candidates.append({
                            'id': parts[0],
                            'name': parts[1],
                            'type': 'pulseaudio_monitor',
                        })

        try:
            pa = pyaudio.PyAudio()
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info['maxInputChannels'] <= 0:
                    continue
                name_lc = info['name'].lower()
                if any(k in name_lc for k in _MONITOR_KEYWORDS):
                    candidates.append({
                        'index': i,
                        'name': info['name'],
                        'type': 'pyaudio_input',
                        'channels': info['maxInputChannels'],
                        'sample_rate': info['defaultSampleRate'],
                    })
            pa.terminate()
        except Exception as exc:  # noqa: BLE001
            print(f'PyAudio scan failed: {exc}')

        return candidates

    # ---------------------------------------------------------------------
    # Probing
    # ---------------------------------------------------------------------

    @staticmethod
    def test_audio_device(device_index: int,
                          rate: int = AUDIO_RATE,
                          channels: int = AUDIO_CHANNELS) -> tuple[bool, float]:
        """Opens, reads from, and closes a device to verify it works.

        Returns:
            Tuple of (success, audio_level_rms).
        """
        try:
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=_PROBE_BUFFER_SIZE,
            )
            data = stream.read(_PROBE_BUFFER_SIZE, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            level = float(np.sqrt(np.mean(samples ** 2)))

            stream.stop_stream()
            stream.close()
            pa.terminate()

            print(f'  Device probe OK — level={level:.1f}')
            return True, level
        except Exception as exc:  # noqa: BLE001
            print(f'  Device probe failed: {exc}')
            return False, 0.0

    # ---------------------------------------------------------------------
    # High-level setup
    # ---------------------------------------------------------------------

    def setup_monitor_device(self) -> bool:
        """Picks and validates a monitor device, populating instance state.

        Returns:
            True if a working device was selected, False otherwise.
        """
        print('\n=== Setting up remote audio capture ===')

        self.list_all_audio_devices()
        candidates = self.find_monitor_devices()
        if not candidates:
            print('No monitor devices found.')
            return False

        print(f'\nFound {len(candidates)} candidate(s):')
        for i, dev in enumerate(candidates):
            print(f"  [{i}] {dev['name']} ({dev.get('type', 'unknown')})")

        selected = self._select_best_candidate(candidates)
        print(f"\nSelected: {selected['name']}")

        if selected['type'] != 'pyaudio_input':
            print('Selected candidate is not a PyAudio input device.')
            return False

        ok, _ = self.test_audio_device(selected['index'])
        if not ok:
            print('All candidate devices failed probing.')
            return False

        self.monitor_device_index = selected['index']
        self.monitor_source = selected['name']
        print(f'Device ready: {self.monitor_source} '
              f'(index={self.monitor_device_index})')
        return True

    @staticmethod
    def _select_best_candidate(candidates: list[dict]) -> dict:
        """Prefers PyAudio inputs whose name suggests a default/pulse source."""
        for dev in candidates:
            if dev['type'] != 'pyaudio_input':
                continue
            name_lc = dev['name'].lower()
            if 'pulse' in name_lc or 'default' in name_lc:
                return dev
        for dev in candidates:
            if dev['type'] == 'pyaudio_input':
                return dev
        return candidates[0]
