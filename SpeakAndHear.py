from asyncio import streams
from speechbrain.pretrained import EncoderDecoderASR
import pyaudio
import speechbrain as sb
import audiofile as af
import sounddevice as sd


# asr_model = EncoderDecoderASR.from_hparams(
#     source="speechbrain/asr-transformer-transformerlm-librispeech",
#     savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
#     run_opts{device="cuda:1"}
# )


def init():
    global asr_model
    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-transformer-transformerlm-librispeech",
        savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
        run_opts={"device": "cuda:1"}
    )


# asr_model = EncoderDecoderASR.from_hparams(
#     source="speechbrain/asr-wav2vec2-commonvoice-en",
#     savedir="pretrained_models/asr-wav2vec2-commonvoice-en",
#     run_opts={"device": "cuda"}
# )


def listen():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1,
                    rate=16000, input=True, frames_per_buffer=1024)
    stream.start_stream()
    print("* Listening mic. Press Ctrl+C to quit...")
    while True:
        try:
            stream.start_stream()
            data = stream.read(4000)
            if len(data) == 0:
                break
            # convert data from bytes to wav file
            wav_file = af.array_to_wave(data, 'temp.wav')
            result = asr_model.transcribe_file(wav_file)
            print(result)
            stream.stop_stream()
        except:  # no microphone
            break


# def record_audio(filename, seconds):
#     fs = 16000
#     print("recording {} ({}s) ...".format(filename, seconds))
#     y = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
#     sd.wait()
#     y = y.T
#     af.write(filename, y, fs)
#     print("  ... saved to {}".format(filename))


def main():
    while True:
        listen()


main()
