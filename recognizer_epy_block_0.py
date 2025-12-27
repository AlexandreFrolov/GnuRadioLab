import numpy as np
from gnuradio import gr
import wave
import subprocess
import tempfile
import os
import threading
import queue
import time
from datetime import datetime
from scipy.signal import butter, lfilter

class blk(gr.sync_block):
    """
    Долговременное непрерывное распознавание через whisper.cpp.
    Результаты пишутся в файлы с разбиением по часам.
    Работает часами, стабильно.
    """

    def __init__(self,
                 whisper_path="C:/whisper/whisper-cli.exe",
                 model_path="C:/whisper/models/ggml-large-v3.bin",
                 sample_rate=48000,
                 buffer_seconds=10,
                 overlap_seconds=5,
                 output_dir="C:/gnuradio_files",
                 num_workers=2):
        gr.sync_block.__init__(self,
                               name="Whisper.cpp Continuous Recognition (Hourly Files)",
                               in_sig=[np.float32],
                               out_sig=None)

        self.whisper_path = whisper_path
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.overlap_samples = int(overlap_seconds * sample_rate)
        self.output_dir = output_dir
        self.buffer = []

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.error_log = os.path.join(self.output_dir, "errors.log")
        self.queue = queue.Queue()
        self.stop_event = threading.Event()

        # Текущий часовой файл и handle
        self.current_hour_file = self.get_hour_file_path()
        self.current_hour_file_handle = open(self.current_hour_file, "a", encoding="utf-8")

        # Запуск пула воркеров
        self.num_workers = num_workers
        for _ in range(self.num_workers):
            threading.Thread(target=self.worker_thread, daemon=True).start()

    def get_hour_file_path(self):
        now = datetime.now()
        filename = f"recognized_{now.strftime('%Y-%m-%d_%H')}.txt"
        return os.path.join(self.output_dir, filename)

    def bandpass_filter(self, data, lowcut=80, highcut=8000, order=5):
        nyq = 0.5 * self.sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    def work(self, input_items, output_items):
        # Ограничение пиков и фильтрация
        audio = np.clip(input_items[0], -1.0, 1.0)
        audio = self.bandpass_filter(audio)
        audio = (audio * 32767).astype(np.int16)

        self.buffer.extend(audio)

        if len(self.buffer) >= self.sample_rate * self.buffer_seconds:
            self.queue.put(self.buffer.copy())
            # сохраняем последние overlap_samples для скользящего окна
            self.buffer = self.buffer[-self.overlap_samples:]

        return len(input_items[0])

    def worker_thread(self):
        while not self.stop_event.is_set():
            try:
                buffer_data = self.queue.get(timeout=1)
                self.process_audio(buffer_data)
            except queue.Empty:
                continue

    def process_audio(self, buffer_data):
        # Создаем временный WAV файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            wav_path = f.name

        try:
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(np.array(buffer_data, dtype=np.int16).tobytes())

            cmd = [
                self.whisper_path,
                "-m", self.model_path,
                "-f", wav_path,
                "-l", "ru",
                "--no-timestamps",
                "--threads", str(os.cpu_count())
            ]

            for attempt in range(3):
                try:
                    si = subprocess.STARTUPINFO()
                    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW

                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        startupinfo=si,
                        text=True,
                        encoding="utf-8",
                        timeout=120
                    )
                    break
                except Exception as e:
                    time.sleep(1)
            else:
                with open(self.error_log, "a", encoding="utf-8") as f:
                    f.write(f"Failed after 3 attempts: {str(e)}\n")
                return

            text = result.stdout.strip()
            if text:
                # Проверяем часовой файл
                now_hour_file = self.get_hour_file_path()
                if now_hour_file != self.current_hour_file:
                    self.current_hour_file_handle.close()
                    self.current_hour_file = now_hour_file
                    self.current_hour_file_handle = open(self.current_hour_file, "a", encoding="utf-8")

                self.current_hour_file_handle.write(text + "\n")
                self.current_hour_file_handle.flush()

        except Exception as e:
            with open(self.error_log, "a", encoding="utf-8") as f:
                f.write(str(e) + "\n")
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    def stop(self):
        self.stop_event.set()
        self.current_hour_file_handle.close()
        return super().stop()
