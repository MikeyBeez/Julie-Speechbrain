from asyncio import streams
from speechbrain.pretrained import EncoderDecoderASR
import pyaudio
import speechbrain as sb
import audiofile as af


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
            if len(data) == 0:
                break
    except KeyboardInterrupt:
        pass
    # convert data from bytes to wav file
    #result = asr_model.transcribe_file("temp.wav")
    # print(result)
    stream.stop_stream()
    stream.close()
    audio.terminate()
    # return result


def play(filename):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=16000, output=True)
    af.play(filename, stream)
    stream.close()
    audio.terminate()


def main():
    init()
    print("* Listening mic. Press Ctrl+C to quit...")
    listen()
    print("* Recording finished")
    play("temp.wav")


main()
