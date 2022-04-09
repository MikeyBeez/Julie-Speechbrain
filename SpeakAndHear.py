from speechbrain.pretrained import EncoderDecoderASR
import speechbrain as sb
import audiofile as af
import sounddevice as sd

# asr_model = EncoderDecoderASR.from_hparams(
#     source="speechbrain/asr-transformer-transformerlm-librispeech",
#     savedir="pretrained_models/asr-transformer-transformerlm-librispeech"
# )


asr_model = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-wav2vec2-commonvoice-en",
    savedir="pretrained_models/asr-wav2vec2-commonvoice-en"
)


def record_audio(filename, seconds):
    fs = 16000
    print("recording {} ({}s) ...".format(filename, seconds))
    y = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    y = y.T
    af.write(filename, y, fs)
    print("  ... saved to {}".format(filename))


def main():
    while True:
        record_audio("temp.wav", 5)
        print(asr_model.transcribe_file("temp.wav"))


main()
