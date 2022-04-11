#############################################################
import subprocess
import pyaudio
import wave
import os
import contextlib


def listen():
    WAVE_OUTPUT_FILENAME = "temp.wav"
    frames = []
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
    stream.start_stream()
    while len(frames) < 40:
        data = stream.read(2000)
        frames.append(data)
        print('length data')
        print(len(data))
        # print the number of frames
        print("* frames: %d" % len(frames))

    stream.stop_stream()
    stream.close()
    p.terminate()
    soundfile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    soundfile.setnchannels(1)
    soundfile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    soundfile.setframerate(16000)
    soundfile.writeframes(b''.join(frames))
    soundfile.close()
    subprocess.call(
        ["sox", "temp.wav", "reversed.wav",
            "silence", "1", "0.1", "1%", "reverse"]
    )
    subprocess.call(
        ["sox", "reversed.wav", "silenced.wav",
            "silence", "1", "0.1", "1%", "reverse"]
    )
    subprocess.call(
        ["sox", "silenced.wav", "temp.wav",
            "noisered", "speech.noiseprofile", "0.3"]
    )

    os.system("play temp.wav")

try:
    while True:
        listen()
except KeyboardInterrupt:
    pass

            
#############################################################