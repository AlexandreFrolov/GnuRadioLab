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
    Долговременное непрерывное распознавание через whisper.cpp
    Без потери данных при закрытии GRC.
    """

    def __init__(self,
                 whisper_path="C:/whisper/whisper-cli.exe",
                 model_path="C:/whisper/models/ggml-large-v3.bin",
                 sample_rate=48000,
                 buffer_seconds=2,
                 output_dir="C:/gnuradio_files"):

        gr.sync_block.__init__(
            self,
            name="Whisper.cpp Continuous Recognition (Safe Shutdown)",
            in_sig=[np.float32],
            out_sig=None
        )

        self.whisper_path = whisper_path
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.output_dir = output_dir

        self.buffer = []
        self.queue = queue.Queue()
        self.stop_event = threading.Event()

        os.makedirs(self.output_dir, exist_ok=True)
        self.error_log = os.path.join(self.output_dir, "errors.log")

        self.current_hour_file = self.get_hour_file_path()

        # НЕ daemon!
        self.worker = threading.Thread(target=self.worker_thread)
        self.worker.start()

    # ------------------------------------------------------------

    def get_hour_file_path(self):
        now = datetime.now()
        return os.path.join(
            self.output_dir,
            f"recognized_{now.strftime('%Y-%m-%d_%H')}.txt"
        )

    # ------------------------------------------------------------

    def work(self, input_items, output_items):
        audio = (input_items[0] * 32767).astype(np.int16)
        self.buffer.extend(audio)

        if len(self.buffer) >= self.sample_rate * self.buffer_seconds:
            self.queue.put(self.buffer.copy())
            self.buffer.clear()

        return len(input_items[0])

    # ------------------------------------------------------------

    def stop(self):
        """
        GNU Radio вызывает stop() при закрытии flowgraph.
        Здесь мы гарантируем, что ничего не потеряется.
        """

        # 1. Отправляем остаток буфера
        if self.buffer:
            self.queue.put(self.buffer.copy())
            self.buffer.clear()

        # 2. Сигнал завершения
        self.stop_event.set()

        # 3. Маркер конца очереди
        self.queue.put(None)

        # 4. Ждём завершения потока
        self.worker.join()

        return True

    # ------------------------------------------------------------

    def worker_thread(self):
        while True:
            try:
                item = self.queue.get(timeout=1)

                if item is None:
                    break

                self.process_audio(item)

            except queue.Empty:
                if self.stop_event.is_set():
                    break

    # ------------------------------------------------------------

    def process_audio(self, buffer_data):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            wav_path = f.name

        try:
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(
                    np.array(buffer_data, dtype=np.int16).tobytes()
                )

            cmd = [
                self.whisper_path,
                "-m", self.model_path,
                "-f", wav_path,
                "-l", "ru",
                "--no-timestamps"
            ]

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
                hour_file = self.get_hour_file_path()
                if hour_file != self.current_hour_file:
                    self.current_hour_file = hour_file

                with open(self.current_hour_file, "a", encoding="utf-8") as f:
                    f.write(text + "\n")
                    f.flush()
                    os.fsync(f.fileno())

        except Exception as e:
            with open(self.error_log, "a", encoding="utf-8") as f:
                f.write(str(e) + "\n")

        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)
