import numpy as np
from gnuradio import gr
import wave
import subprocess
import tempfile
import os

class blk(gr.sync_block):
    """
    Embedded Python Block for offline Russian speech recognition
    using whisper.cpp (any model), output to UTF-8 file.
    Fully silent, avoids console output and blocking.
    """

    def __init__(self,
                 whisper_path="C:/whisper/whisper-cli.exe",
                 model_path="C:/whisper/models/ggml-large-v3.bin",
                 sample_rate=48000,
                 buffer_seconds=2,
                 output_dir="C:/gnuradio_files"):
        gr.sync_block.__init__(
            self,
            name="Whisper.cpp Speech Recognition (File Output, Silent)",
            in_sig=[np.float32],
            out_sig=None  # убрали выход
        )

        self.whisper_path = whisper_path
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.output_dir = output_dir
        self.buffer = []

        # Проверка путей
        if not os.path.exists(self.whisper_path):
            raise RuntimeError(f"whisper-cli.exe not found at {self.whisper_path}")
        if not os.path.exists(self.model_path):
            raise RuntimeError(f"Whisper model not found at {self.model_path}")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.output_file = os.path.join(self.output_dir, "recognized.txt")
        self.error_log = os.path.join(self.output_dir, "errors.log")

    def work(self, input_items, output_items):
        # Копим аудио в буфер
        audio = (input_items[0] * 32767).astype(np.int16)
        self.buffer.extend(audio)

        # Если накопилось buffer_seconds аудио, запускаем распознавание
        if len(self.buffer) >= self.sample_rate * self.buffer_seconds:
            self.process_audio()
            self.buffer = []

        return len(input_items[0])

    def process_audio(self):
        # Создаём временный WAV-файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            wav_path = f.name

        try:
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(np.array(self.buffer, dtype=np.int16).tobytes())

            # Запуск whisper-cli.exe без консольного окна
            cmd = [
                self.whisper_path,
                "-m", self.model_path,
                "-f", wav_path,
                "-l", "ru",
                "--no-timestamps"
            ]

            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW,
                text=True,
                encoding="utf-8"
            )

            # Читаем и игнорируем вывод (предотвращаем подвисание)
            stdout, stderr = p.communicate(timeout=60)

            # Записываем результат в UTF-8 файл
            text = stdout.strip()
            if text:
                with open(self.output_file, "a", encoding="utf-8") as f:
                    f.write(text + "\n")

        except Exception as e:
            # Лог ошибок
            with open(self.error_log, "a", encoding="utf-8") as f:
                f.write(str(e) + "\n")

        finally:
            # Удаляем временный WAV
            if os.path.exists(wav_path):
                os.remove(wav_path)
