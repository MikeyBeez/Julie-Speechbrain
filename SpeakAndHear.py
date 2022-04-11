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
    # initialize filter variables

    # fir = np.zeros(CHUNK * 2)
    # fir[:(2 * CHUNK)] = 1.
    # fir /= fir.sum()

    # fir_last = fir
    # avg_freq_buffer = np.zeros(CHUNK)
    # obj = -np.inf
    # t = 10

    # initialize sample buffer
    buffer = np.zeros(CHUNK * 2)
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


def crap():
    #         strings_audio_data = stream.read(1024)
    #         audio_data = np.fromstring(strings_audio_data, dtype=np.int16)
    #         normalized_data = audio_data / 32768.0
    #         frequency_data = np.fft.rfft(normalized_data)

    #         # synthesize audio
    #         buffer[1024:] = np.random.randn(1024)
    #         freq_buffer = np.fft.fft(buffer)
    #         freq_fir = np.fft.fft(fir)
    #         freq_synth = freq_fir * freq_buffer
    #         synth = np.real(np.fft.ifft(freq_synth))

    #         # adjust fir
    #         # objective is to make abs(freq_synth) as much like long-term average of freq_buffer
    #         MEMORY = 100
    #         avg_freq_buffer = (avg_freq_buffer * MEMORY +
    #                            np.abs(freq_data)) / (MEMORY + 1)
    #         obj_last = obj

    #         obj = np.real(np.dot(avg_freq_buffer[1:51], np.abs(freq_synth[1:100:2])) / np.dot(freq_synth[1:100:2],
    #                                                                                           np.conj(freq_synth[1:100:2])))
    #         if obj > obj_last:
    #             fir_last = fir
    #         fir = fir_last.copy()

    #         # adjust filter in frequency space
    #         freq_fir = np.fft.fft(fir)
    #         # t += np.clip(np.random.randint(3)-1, 0, 64)
    #         t = np.random.randint(100)
    #         freq_fir[t] = np.real(freq_fir[t])
    #         freq_fir[t] += np.random.randn() * .05

    #         # transform frequency space filter to time space, click-free
    #         fir = np.real(np.fft.ifft(freq_fir))
    #         fir[:CHUNK] *= np.linspace(1., 0., CHUNK) ** .1  # type: ignore
    #         fir[CHUNK:] = 0

    #         # move chunk to start of buffer
    #         buffer[:CHUNK] = buffer[CHUNK:]
    #         # write audio
    #         audio_data = np.array(np.round_(synth[CHUNK:] * MAX_INT), dtype=DTYPE)
    #         string_audio_data = audio_data.tostring()
    #         stream.write(string_audio_data, CHUNK)
    pass


def main():
    init()
    print("* Listening mic. Press Ctrl+C to quit...")
    listen()
    print("* Recording finished")
    WAVE_OUTPUT_FILENAME = "temp.wav"
    os.system("aplay " + WAVE_OUTPUT_FILENAME)


main()
