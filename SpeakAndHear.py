from speechbrain.pretrained import EncoderDecoderASR
import speechbrain as sb
import audiofile as af
import sounddevice as sd

asr_model = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-crdnn-rnnlm-librispeech",
    savedir="pretrained_model")


def record_audio(filename, seconds):
    fs = 16000
    print("recording {} ({}s) ...".format(filename, seconds))
    y = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    y = y.T
    af.write(filename, y, fs)
    print("  ... saved to {}".format(filename))


def listen(seconds):
    record_audio("temp.wav", seconds)
    print(asr_model.transcribe_file("temp.wav"))


def main():
    while True:
        listen(3)
