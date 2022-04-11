from asyncio import streams
from gettext import npgettext
from re import I
from speechbrain.pretrained import EncoderDecoderASR
import pyaudio
import audiofile as af
import os
import wave
import numpy as np
import scipy.signal as signal


def init():
    global asr_model
    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-transformer-transformerlm-librispeech",
        savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
        run_opts={"device": "cuda:1"}
    )


def listen():
    WAVE_OUTPUT_FILENAME = "temp.wav"
    CHUNK = 1024
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=16000, input=True, frames_per_buffer=1024
                        )
    frames = []
    try:
        while True:
            data = stream.read(1024)
            frames.append(data)
            if len(data) == 0:
                break
    except KeyboardInterrupt:
        pass
    stream.stop_stream()
    stream.close()
    audio.terminate()
    soundfile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    soundfile.setnchannels(1)
    soundfile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    soundfile.setframerate(16000)
    soundfile.writeframes(b''.join(frames))
    soundfile.close()


def main():
    init()
    print("* Listening mic. Press Ctrl+C to quit...")
    listen()
    print("* Recording finished")
    WAVE_OUTPUT_FILENAME = "temp.wav"
    os.system("aplay " + WAVE_OUTPUT_FILENAME)


main()
