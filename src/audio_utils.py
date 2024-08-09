import pyaudio
import librosa
import numpy as np
def select_input_device():
    p = pyaudio.PyAudio()
    devices = []
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        devices.append(device_info['name'])
    p.terminate()

    print("Available Input Devices:")
    for i, device in enumerate(devices):
        print(f"{i + 1}. {device}")

    selected_index = int(input("Enter the index of the input device you want to use: ")) - 1
    return selected_index
