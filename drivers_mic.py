import sounddevice as sd

devices = sd.query_devices()
for i, dev in enumerate(devices):
    if dev['max_input_channels'] > 0:
        print(f"{i}: {dev['name']} - {dev['max_input_channels']} input channels")
