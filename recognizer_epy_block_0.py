import numpy as np
from gnuradio import gr
import wave
import subprocess
import tempfile
import os
import threading
import queue
from datetime import datetime

class blk(gr.sync_block):
    """
    Долговременное непрерывное распознавание через whisper.cpp.
    Результаты пишутся в файлы с разбиением по часам.
    Консоль чистая, работает часами.
    """

    def __init__(self,
                 whisper_path="C:/whisper/whisper-cli.exe",
                 model_path="C:/whisper/models/ggml-large-v3.bin",
                 sample_rate=48000,
                 buffer_seconds=2,
                 output_dir="C:/gnuradio_files"):
        gr.sync_block.__init__(self,
                               name="Whisper.cpp Continuous Recognition (Hourly Files)",
                               in_sig=[np.float32],
                               out_sig=None)  # Нет выхода

        self.whisper_path = whisper_path
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.output_dir = output_dir
        self.buffer = []

        # Проверка директорий
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.error_log = os.path.join(self.output_dir, "errors.log")
        self.queue = queue.Queue()
        self.stop_event = threading.Event()

        # Текущий часовой файл
        self.current_hour_file = self.get_hour_file_path()

        # Запуск фонового потока
        threading.Thread(target=self.worker_thread, daemon=True).start()

    def get_hour_file_path(self):
        now = datetime.now()
        filename = f"recognized_{now.strftime('%Y-%m-%d_%H')}.txt"
        return os.path.join(self.output_dir, filename)

    def work(self, input_items, output_items):
        audio = (input_items[0] * 32767).astype(np.int16)
        self.buffer.extend(audio)

        if len(self.buffer) >= self.sample_rate * self.buffer_seconds:
            # Копируем буфер и отправляем в очередь
            self.queue.put(self.buffer.copy())
            self.buffer = []

        return len(input_items[0])

    def worker_thread(self):
        while not self.stop_event.is_set():
            try:
                buffer_data = self.queue.get(timeout=1)
                self.process_audio(buffer_data)
            except queue.Empty:
                continue

    def process_audio(self, buffer_data):
        # Создаем временный WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            wav_path = f.name

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
            "--no-timestamps"
        ]

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

            text = result.stdout.strip()
            if text:
                # Обновляем текущий часовой файл
                hour_file = self.get_hour_file_path()
                if hour_file != self.current_hour_file:
                    self.current_hour_file = hour_file

                with open(self.current_hour_file, "a", encoding="utf-8") as f:
                    f.write(text + "\n")

        except Exception as e:
            with open(self.error_log, "a", encoding="utf-8") as f:
                f.write(str(e) + "\n")
        finally:
            os.remove(wav_path)
