from abc import ABC
import types
import speech_recognition as sr


class Microphone(sr.Microphone, ABC):
    def __init__(self, device_index=None, sample_rate=None, chunk_size=1024):
        types.MethodType(self.get_pyaudio, super())
        #sr.Microphone.get_pyaudio = self.get_pyaudio
        super().__init__(device_index=device_index, sample_rate=sample_rate, chunk_size=chunk_size)

    def __enter__(self):
        assert self.stream is None, "This audio source is already inside a context manager"
        self.audio = self.pyaudio_module.PyAudio()
        try:
            device = self.audio.get_device_info_by_index(self.device_index)
            print(f"Using Device: {device['name']} (defaultSampleRate={device['defaultSampleRate']}, maxInputChannels={device['maxInputChannels']})")
            # channels = self.audio.get_device_info_by_index(self.device_index)['maxInputChannels']
            channels = 1
            is_input = True

            # @todo: Fix for "OSError: [Errno -9997] Invalid sample rate" with loopback devices
            #if device['isLoopbackDevice']:
            #    self.SAMPLE_RATE = int(device['defaultSampleRate'])

            self.stream = sr.Microphone.MicrophoneStream(
                self.audio.open(
                    input_device_index=self.device_index, channels=channels,
                    format=self.format, rate=self.SAMPLE_RATE, frames_per_buffer=self.CHUNK,
                    input=is_input,  # stream is an input stream
                    # output=not is_input
                )
            )
        except Exception:
            self.audio.terminate()
            raise
        return self

    @staticmethod
    def get_pyaudio():
        """
        Imports the pyaudio module and checks its version. Throws exceptions if pyaudio can't be found or a wrong version is installed
        """
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            raise AttributeError("Could not find PyAudio; check installation")
        from distutils.version import LooseVersion
        if LooseVersion(pyaudio.__version__) < LooseVersion("0.2.11"):
            raise AttributeError("PyAudio 0.2.11 or later is required (found version {})".format(pyaudio.__version__))
        return pyaudio


class Recognizer(sr.Recognizer, ABC):
    __init__ = sr.Recognizer.__init__
