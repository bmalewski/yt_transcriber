import sys
import os
import glob
import subprocess
import time
import requests
import traceback
from multiprocessing import Process
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit,
                               QComboBox, QCheckBox, QPushButton, QProgressBar,
                               QVBoxLayout, QGridLayout, QMessageBox, QFileDialog,
                               QPlainTextEdit, QSizePolicy)
from PySide6.QtCore import Qt, QThread, Signal
import yt_dlp
from docx import Document
from openai import OpenAI

# -------------------------------------------------
# Ścieżki
# -------------------------------------------------
base_path = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
downloads_dir = os.path.join(base_path, "downloads")
os.makedirs(downloads_dir, exist_ok=True)

server_process = None

# -------------------------------------------------
# API Keys
# -------------------------------------------------
def load_api_key(file_name):
    path = os.path.join(base_path, file_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

def save_api_key(file_name, key):
    path = os.path.join(base_path, file_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(key.strip())

openai_api_key = load_api_key("openai_api_key.txt")

# -------------------------------------------------
# Serwer Faster-Whisper (zintegrowany)
# -------------------------------------------------
def run_faster_server():
    import tempfile
    from fastapi import FastAPI, UploadFile, Form
    import uvicorn
    from faster_whisper import WhisperModel

    app = FastAPI()
    loaded_models = {}

    @app.post("/transcribe")
    async def transcribe(file: UploadFile, model: str = Form("small"), device: str = Form("cpu")):
        if (model, device) not in loaded_models:
            loaded_models[(model, device)] = WhisperModel(model, device=device, compute_type="int8")
        whisper_model = loaded_models[(model, device)]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        segments, info = whisper_model.transcribe(tmp_path)
        text, result_segments = "", []
        for seg in segments:
            seg_dict = {"start": seg.start, "end": seg.end, "text": seg.text}
            result_segments.append(seg_dict)
            text += seg.text.strip() + " "

        os.remove(tmp_path)
        return {"text": text.strip(), "segments": result_segments}

    uvicorn.run(app, host="127.0.0.1", port=8000)

def start_faster_server():
    global server_process
    if server_process is None or not server_process.is_alive():
        server_process = Process(target=run_faster_server, daemon=True)
        server_process.start()
        for _ in range(20):
            try:
                r = requests.get("http://127.0.0.1:8000/docs")
                if r.status_code == 200:
                    return
            except:
                time.sleep(0.5)
        raise RuntimeError("Serwer faster-whisper nie uruchomił się w czasie.")

def transcribe_with_faster_whisper(audio_path, model_name="small", device="cpu"):
    url = "http://127.0.0.1:8000/transcribe"
    with open(audio_path, "rb") as f:
        r = requests.post(url, files={"file": f}, data={"model": model_name, "device": device})
    r.raise_for_status()
    return r.json()

# -------------------------------------------------
# Formatowanie i zapisywanie plików
# -------------------------------------------------
def format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds * 1000) % 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def save_txt(text, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def save_docx(text, path):
    doc = Document()
    doc.add_paragraph(text)
    doc.save(path)

def save_html(text, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>\n")
        for line in text.splitlines():
            f.write(f"<p>{line}</p>\n")
        f.write("</body></html>")

def save_srt(segments, path):
    if not segments:
        return
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = format_timestamp(seg.get("start", 0))
            end = format_timestamp(seg.get("end", seg.get("start", 0)))
            text = seg.get("text", "").strip()
            if not text:
                continue
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")

# -------------------------------------------------
# Wątek transkrypcji
# -------------------------------------------------
class TranscriptionThread(QThread):
    progress_signal = Signal(int)
    status_signal = Signal(str)
    finished_signal = Signal(str)

    def __init__(self, url, local_file, transcription_model, whisper_variant,
                 translation_model, device, src_lang_code, tgt_lang_code, formats, openai_key):
        super().__init__()
        self.url = url
        self.local_file = local_file
        self.transcription_model = transcription_model
        self.whisper_variant = whisper_variant
        self.translation_model = translation_model
        self.device = device
        self.src_lang_code = src_lang_code
        self.tgt_lang_code = tgt_lang_code
        self.formats = formats
        self.openai_key = openai_key

    def run(self):
        try:
            if self.local_file:
                audio_path = self.local_file
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                self.status_signal.emit(f"Używam lokalnego pliku: {audio_path}")
            else:
                self.status_signal.emit("Pobieranie audio...")
                audio_path, base_name = self.download_audio(self.url)

            segments = []
            text = ""

            client = None
            if self.openai_key:
                client = OpenAI(api_key=self.openai_key)

            if self.translation_model == "Brak":
                if self.transcription_model == "Faster-Whisper (lokalny)":
                    self.status_signal.emit("Uruchamianie serwera Faster-Whisper...")
                    start_faster_server()
                    self.status_signal.emit(f"Transkrypcja (Faster-Whisper, {self.whisper_variant})...")
                    result = transcribe_with_faster_whisper(audio_path, model_name=self.whisper_variant, device=self.device)
                    segments = result.get("segments", [])
                    text = result.get("text", "")
                else:
                    with open(audio_path, "rb") as f:
                        self.status_signal.emit("Transkrypcja online (OpenAI)...")
                        transcript = client.audio.transcriptions.create(
                            model="gpt-4o-mini-transcribe",
                            file=f
                        )
                        text = transcript.text

            elif self.translation_model == "OpenAI (ASR+ST)":
                self.status_signal.emit("Transkrypcja + tłumaczenie online (OpenAI)...")
                with open(audio_path, "rb") as f:
                    translation = client.audio.translations.create(
                        model="gpt-4o-mini-transcribe",
                        file=f
                    )
                    text = translation.text

            self.status_signal.emit("Zapisywanie plików...")
            for i, fmt in enumerate(self.formats, start=1):
                percent = int(80 + (i / len(self.formats)) * 20)
                self.progress_signal.emit(percent)
                path = os.path.join(downloads_dir, f"{base_name}.{fmt}")
                if fmt == "txt":
                    save_txt(text, path)
                elif fmt == "docx":
                    save_docx(text, path)
                elif fmt == "html":
                    save_html(text, path)
                elif fmt == "srt" and segments:
                    save_srt(segments, path)

            self.progress_signal.emit(100)
            self.finished_signal.emit(f"Zakończono. Pliki zapisane w {downloads_dir}")

        except Exception as e:
            self.finished_signal.emit(f"Błąd: {str(e)}\n{traceback.format_exc()}")

    def download_audio(self, url):
        video_info = {}
        def hook(d):
            status = d.get("status")
            if status == 'downloading':
                percent_str = d.get('_percent_str', '0%').strip()
                try:
                    percent = float(percent_str.replace('%',''))
                except:
                    percent = 0
                self.progress_signal.emit(int(percent))
                self.status_signal.emit(f"Pobieranie audio... {percent_str}")
            elif status == 'finished':
                video_info['filename'] = d.get('filename')
                video_info['title'] = d.get('info_dict', {}).get('title', video_info.get('title'))

        output_template = os.path.join(downloads_dir, "%(title)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "progress_hooks": [hook],
            "postprocessors": [dict(key="FFmpegExtractAudio", preferredcodec="mp3", preferredquality="192")],
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        candidate = None
        fname = video_info.get('filename')
        if fname and os.path.exists(fname):
            candidate = fname
        else:
            mp3s = glob.glob(os.path.join(downloads_dir, "*.mp3"))
            if mp3s:
                candidate = sorted(mp3s, key=os.path.getmtime, reverse=True)[0]

        if not candidate or not os.path.exists(candidate):
            raise FileNotFoundError("Nie udało się pobrać pliku audio.")

        base_name = os.path.splitext(os.path.basename(candidate))[0]
        return candidate, base_name

# -------------------------------------------------
# GUI
# -------------------------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube / Plik → Transkrypcja / Tłumaczenie")
        self.setMinimumWidth(700)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        grid = QGridLayout()
        self.layout.addLayout(grid)

        grid.addWidget(QLabel("Adres URL:"), 0, 0)
        self.url_entry = QLineEdit()
        grid.addWidget(self.url_entry, 0, 1)

        grid.addWidget(QLabel("Plik lokalny:"), 1, 0)
        self.file_btn = QPushButton("Wybierz plik…")
        self.file_btn.clicked.connect(self.choose_file)
        grid.addWidget(self.file_btn, 1, 1)
        self.local_file = None

        grid.addWidget(QLabel("Model transkrypcji:"), 2, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Faster-Whisper (lokalny)", "OpenAI (online)"])
        grid.addWidget(self.model_combo, 2, 1)

        grid.addWidget(QLabel("Wariant Whispera:"), 3, 0)
        self.variant_combo = QComboBox()
        self.variant_combo.addItems(["tiny", "base", "small", "medium", "large-v2", "large-v3"])
        grid.addWidget(self.variant_combo, 3, 1)

        grid.addWidget(QLabel("Model tłumaczenia:"), 4, 0)
        self.translation_combo = QComboBox()
        self.translation_combo.addItems(["Brak", "OpenAI (ASR+ST)"])
        grid.addWidget(self.translation_combo, 4, 1)

        grid.addWidget(QLabel("Urządzenie:"), 5, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])
        grid.addWidget(self.device_combo, 5, 1)

        grid.addWidget(QLabel("Język źródłowy:"), 6, 0)
        self.source_lang_combo = QComboBox()
        self.source_lang_combo.addItems(["angielski", "niemiecki", "francuski", "hiszpański", "włoski", "polski"])
        grid.addWidget(self.source_lang_combo, 6, 1)

        grid.addWidget(QLabel("Język docelowy:"), 7, 0)
        self.target_lang_combo = QComboBox()
        self.target_lang_combo.addItems(["polski", "angielski", "niemiecki", "francuski", "hiszpański", "włoski"])
        grid.addWidget(self.target_lang_combo, 7, 1)

        grid.addWidget(QLabel("OpenAI API Key:"), 8, 0)
        self.apikey_entry = QLineEdit()
        self.apikey_entry.setEchoMode(QLineEdit.Password)
        self.apikey_entry.setText(openai_api_key)
        grid.addWidget(self.apikey_entry, 8, 1)

        grid.addWidget(QLabel("Formaty zapisu:"), 9, 0)
        self.txt_cb = QCheckBox("TXT"); self.txt_cb.setChecked(True); grid.addWidget(self.txt_cb, 9, 1)
        self.docx_cb = QCheckBox("DOCX"); self.docx_cb.setChecked(True); grid.addWidget(self.docx_cb, 10, 1)
        self.srt_cb = QCheckBox("SRT"); self.srt_cb.setChecked(True); grid.addWidget(self.srt_cb, 11, 1)
        self.html_cb = QCheckBox("HTML"); grid.addWidget(self.html_cb, 12, 1)

        self.start_btn = QPushButton("Start")
        self.start_btn.setFixedWidth(120)
        self.start_btn.clicked.connect(self.start_transcription)
        self.layout.addWidget(self.start_btn, alignment=Qt.AlignCenter)

        # Okno logów (terminal)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.layout.addWidget(self.log_box)
        self.append_log("== Aplikacja uruchomiona ==")

        # Pasek postępu
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.progress_bar)

    def append_log(self, text):
        self.log_box.appendPlainText(text)

    def choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Wybierz plik audio/wideo", "", "Audio/Video Files (*.mp3 *.wav *.mp4 *.mkv *.avi)")
        if file_path:
            self.local_file = file_path
            self.url_entry.clear()
            self.append_log(f"Wybrano plik lokalny: {file_path}")

    def start_transcription(self):
        url = self.url_entry.text().strip()
        if not url and not self.local_file:
            QMessageBox.critical(self, "Błąd", "Podaj URL albo wybierz plik lokalny")
            return
        formats = []
        if self.txt_cb.isChecked(): formats.append("txt")
        if self.docx_cb.isChecked(): formats.append("docx")
        if self.srt_cb.isChecked(): formats.append("srt")
        if self.html_cb.isChecked(): formats.append("html")
        if not formats:
            QMessageBox.critical(self, "Błąd", "Wybierz przynajmniej jeden format pliku")
            return

        src_lang_map = {
            "angielski": "en",
            "niemiecki": "de",
            "francuski": "fr",
            "hiszpański": "es",
            "włoski": "it",
            "polski": "pl"
        }
        source_lang_code = src_lang_map[self.source_lang_combo.currentText()]

        tgt_lang_map = {
            "polski": "pl",
            "angielski": "en",
            "niemiecki": "de",
            "francuski": "fr",
            "hiszpański": "es",
            "włoski": "it"
        }
        target_lang_code = tgt_lang_map[self.target_lang_combo.currentText()]

        openai_key = self.apikey_entry.text().strip()
        if openai_key:
            save_api_key("openai_api_key.txt", openai_key)

        device = self.device_combo.currentText()

        self.start_btn.setEnabled(False)
        self.thread = TranscriptionThread(
            url=url,
            local_file=self.local_file,
            transcription_model=self.model_combo.currentText(),
            whisper_variant=self.variant_combo.currentText(),
            translation_model=self.translation_combo.currentText(),
            device=device,
            src_lang_code=source_lang_code,
            tgt_lang_code=target_lang_code,
            formats=formats,
            openai_key=openai_key
        )
        self.thread.progress_signal.connect(self.progress_bar.setValue)
        self.thread.status_signal.connect(self.append_log)
        self.thread.finished_signal.connect(self.on_finished)
        self.thread.start()

    def on_finished(self, msg):
        self.append_log(msg)
        QMessageBox.information(self, "Gotowe", msg)
        self.start_btn.setEnabled(True)

# -------------------------------------------------
# Start
# -------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QWidget {
            background-color: #090040;
            color: #FFFFFF;
            font-size: 11pt;
            font-weight: bold;
        }
        QLineEdit, QComboBox, QPushButton {
            background-color: #471396;
            border: 1px solid #B13BFF;
            border-radius: 4px;
            padding: 4px;
        }
        QPushButton:hover {
            background-color: #B13BFF;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border-radius: 4px;
            background: #B13BFF;
        }
        QCheckBox::indicator:checked {
            background-color: #FFCC00;
        }
        QProgressBar {
            border: 1px solid #FFFFFF;
            border-radius: 5px;
            text-align: center;
            color: #FFFFFF;
            font-weight: bold;
        }
        QProgressBar::chunk {
            background-color: #FFCC00;
            border-radius: 5px;
        }
        QPlainTextEdit {
            background-color: #000000;
            color: #FFCC00;
            font-family: Consolas, monospace;
            font-size: 11pt;
            font-weight: bold;
        }
    """)
    window = MainWindow()
    window.show()
    exit_code = app.exec()

    if server_process is not None and server_process.is_alive():
        server_process.terminate()

    sys.exit(exit_code)