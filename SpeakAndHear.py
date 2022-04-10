from asyncio import streams
from speechbrain.pretrained import EncoderDecoderASR
import pyaudio
import speechbrain as sb
import audiofile as af
import os
import wave


def init():
    global WAVE_OUTPUT_FILENAME
    WAVE_OUTPUT_FILENAME = "temp.wav"
    global asr_model
    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-transformer-transformerlm-librispeech",
        savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
        run_opts={"device": "cuda"}
    )


def listen():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=16000, input=True, frames_per_buffer=1024
                        )
    frames = []
    try:
        while True:
            data = stream.read(1024)
            frames.append(data)
            # if len(data) == 0:
            #     break
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


# play temp.wav file


def play(filename):
    os.system("aplay " + filename)


def main():
    init()
    print("* Listening mic. Press Ctrl+C to quit...")
    listen()
    print("* Recording finished")
    play("temp.wav")


main()
