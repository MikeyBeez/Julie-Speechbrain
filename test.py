# use speechbrain to transcribe from microphone

def main():
    from speechbrain.pretrained import EncoderDecoderASR
    import speechbrain as sb
    import audiofile as af
    import sounddevice as sd
    import os
    import time
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Speak and hear')
    parser.add_argument('--source', type=str, default='speechbrain/asr-crdnn-rnnlm-librispeech',
                        help='source model')
    parser.add_argument('--savedir', type=str, default='pretrained_model',
                        help='savedir')
    parser.add_argument('--seconds', type=float, default=3,
                        help='seconds to record')
    parser.add_argument('--output', type=str, default='output.wav',
                        help='output file')
    args = parser.parse_args()

    asr_model = EncoderDecoderASR.from_hparams(
        source=args.source,
        savedir=args.savedir)

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

    def transcribe(seconds):
        record_audio("temp.wav", seconds)
        print(asr_model.transcribe_file("temp.wav"))

    def main():
        while True:
            listen(3)
            # print(asr_model.transcribe_file("temp.wav"))

    main()
