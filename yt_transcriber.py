import sys
import os
import re
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit,
                               QComboBox, QCheckBox, QPushButton, QProgressBar,
                               QFileDialog, QVBoxLayout, QGridLayout, QMessageBox)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QIcon
import yt_dlp
import whisper
from docx import Document
from openai import OpenAI

# -------------------------------------------------
# Ścieżka do folderu programu i assets
# -------------------------------------------------
base_path = os.path.dirname(sys.executable) if getattr(sys,'frozen',False) else os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(base_path, "whisper", "assets")

# -------------------------------------------------
# Wątek do pobierania modelu Whispera
# -------------------------------------------------
class ModelDownloadThread(QThread):
    progress_signal = Signal(int)
    status_signal = Signal(str)
    finished_signal = Signal(object)  # zwróci model

    def __init__(self, model_name="small", device="cpu"):
        super().__init__()
        self.model_name = model_name
        self.device = device

    def run(self):
        if not os.path.exists(assets_path):
            os.makedirs(assets_path)
        self.status_signal.emit(f"Pobieranie modelu {self.model_name}...")
        try:
            model = whisper.load_model(self.model_name, device=self.device, download_root=assets_path)
            self.finished_signal.emit(model)
        except Exception as e:
            self.finished_signal.emit(None)

# -------------------------------------------------
# Funkcje zapisu plików
# -------------------------------------------------
def save_txt(text, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def save_docx(text, path):
    doc = Document()
    doc.add_paragraph(text)
    doc.save(path)

def format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds * 1000) % 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def save_srt(segments, path):
    if not segments:
        return
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = seg["start"]
            end = seg["end"]
            text = seg["text"].strip()
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")

def save_html(text, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("<html><body>\n")
        for line in text.splitlines():
            f.write(f"<p>{line}</p>\n")
        f.write("</body></html>")

# -------------------------------------------------
# Wątek do pobierania audio i transkrypcji
# -------------------------------------------------
class TranscriptionThread(QThread):
    progress_signal = Signal(int)
    status_signal = Signal(str)
    finished_signal = Signal(str)

    def __init__(self, url, model_name, device, backend, translate, formats, folder):
        super().__init__()
        self.url = url
        self.model_name = model_name
        self.device = device
        self.backend = backend
        self.translate = translate
        self.formats = formats
        self.folder = folder

    def run(self):
        try:
            self.status_signal.emit("Pobieranie audio...")
            audio_path = self.download_audio(self.url)

            if self.backend == "Lokalny":
                text, segments = self.transcribe_local(audio_path)
            else:
                text, segments = self.transcribe_cloud(audio_path)

            # zapis plików
            for fmt in self.formats:
                path = os.path.join(self.folder, f"output.{fmt}")
                if fmt == "txt":
                    save_txt(text, path)
                elif fmt == "docx":
                    save_docx(text, path)
                elif fmt == "srt":
                    save_srt(segments, path)
                elif fmt == "html":
                    save_html(text, path)

            self.finished_signal.emit(f"Zakończono. Pliki zapisane w {self.folder}")

        except Exception as e:
            self.finished_signal.emit(f"Błąd: {str(e)}")

    def download_audio(self, url):
        output_template = "audio"
        def hook(d):
            if d['status'] == 'downloading':
                percent_str = d.get('_percent_str', '0%').strip()
                clean_str = re.sub(r'\x1b\[[0-9;]*m', '', percent_str)
                try:
                    percent = float(clean_str.replace('%',''))
                except:
                    percent = 0
                self.progress_signal.emit(int(percent))
                self.status_signal.emit(f"Pobieranie audio... {percent_str}")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "progress_hooks": [hook],
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_template + ".mp3"

    def transcribe_local(self, audio_path):
        self.status_signal.emit("Transkrypcja lokalna...")
        model = whisper.load_model(self.model_name, device=self.device, download_root=assets_path)
        task = "translate" if self.translate else "transcribe"
        result = model.transcribe(audio_path, task=task)
        return result["text"], result.get("segments")

    def transcribe_cloud(self, audio_path):
        self.status_signal.emit("Transkrypcja w chmurze...")
        client = OpenAI()
        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f
            )
        text = transcript.text
        return text, None

# -------------------------------------------------
# GUI PySide6
# -------------------------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube → Transkrypcja / Tłumaczenie")
        self.setMinimumWidth(500)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        grid = QGridLayout()
        self.layout.addLayout(grid)

        # URL
        grid.addWidget(QLabel("Adres URL:"),0,0)
        self.url_entry = QLineEdit()
        grid.addWidget(self.url_entry,0,1)

        # Model
        grid.addWidget(QLabel("Model Whisper:"),1,0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny","base","small","medium","large"])
        grid.addWidget(self.model_combo,1,1)

        # Urządzenie
        grid.addWidget(QLabel("Urządzenie:"),2,0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu","cuda"])
        grid.addWidget(self.device_combo,2,1)

        # Backend
        grid.addWidget(QLabel("Backend:"),3,0)
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["Lokalny","Chmura"])
        grid.addWidget(self.backend_combo,3,1)

        # Tłumaczenie
        self.translate_cb = QCheckBox("Tłumaczenie na polski")
        self.translate_cb.setIcon(QIcon("icons/translate.png"))
        grid.addWidget(self.translate_cb,4,0,1,2)

        # Format plików
        grid.addWidget(QLabel("Formaty zapisu:"),5,0)
        self.txt_cb = QCheckBox("TXT"); self.txt_cb.setIcon(QIcon("icons/txt.png")); self.txt_cb.setChecked(True); grid.addWidget(self.txt_cb,5,1)
        self.docx_cb = QCheckBox("DOCX"); self.docx_cb.setIcon(QIcon("icons/docx.png")); self.docx_cb.setChecked(True); grid.addWidget(self.docx_cb,6,1)
        self.srt_cb = QCheckBox("SRT"); self.srt_cb.setIcon(QIcon("icons/srt.png")); self.srt_cb.setChecked(True); grid.addWidget(self.srt_cb,7,1)
        self.html_cb = QCheckBox("HTML"); self.html_cb.setIcon(QIcon("icons/html.png")); grid.addWidget(self.html_cb,8,1)

        # Ikony w menu rozwijanym
        self.device_combo.setItemIcon(0, QIcon("icons/cpu.png"))
        self.device_combo.setItemIcon(1, QIcon("icons/gpu.png"))
        self.backend_combo.setItemIcon(0, QIcon("icons/local.png"))
        self.backend_combo.setItemIcon(1, QIcon("icons/cloud.png"))
        for i, name in enumerate(["tiny","base","small","medium","large"]):
            self.model_combo.setItemIcon(i, QIcon(f"icons/{name}.png"))

        # Pasek postępu i status
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)
        self.status_label = QLabel("")
        self.layout.addWidget(self.status_label)

        # Start
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_transcription)
        self.layout.addWidget(self.start_btn)

    def start_transcription(self):
        url = self.url_entry.text().strip()
        if not url:
            QMessageBox.critical(self,"Błąd","Podaj URL filmu z YouTube")
            return
        formats = []
        if self.txt_cb.isChecked(): formats.append("txt")
        if self.docx_cb.isChecked(): formats.append("docx")
        if self.srt_cb.isChecked(): formats.append("srt")
        if self.html_cb.isChecked(): formats.append("html")
        if not formats:
            QMessageBox.critical(self,"Błąd","Wybierz przynajmniej jeden format pliku")
            return
        folder = QFileDialog.getExistingDirectory(self,"Wybierz folder do zapisu")
        if not folder: return

        # Uruchom wątek transkrypcji
        self.thread = TranscriptionThread(
            url=url,
            model_name=self.model_combo.currentText(),
            device=self.device_combo.currentText(),
            backend=self.backend_combo.currentText(),
            translate=self.translate_cb.isChecked(),
            formats=formats,
            folder=folder
        )
        self.thread.progress_signal.connect(self.progress_bar.setValue)
        self.thread.status_signal.connect(self.status_label.setText)
        self.thread.finished_signal.connect(lambda msg: QMessageBox.information(self,"Gotowe",msg))
        self.thread.start()

# -------------------------------------------------
# Start aplikacji z ciemnym motywem
# -------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyleSheet("""
        QWidget {
            background-color: #2b2b2b;
            color: #f0f0f0;
            font-family: 'Segoe UI', sans-serif;
            font-size: 12pt;
        }
        QLineEdit, QComboBox, QCheckBox {
            background-color: #3c3f41;
            border: 1px solid #555;
            padding: 4px;
            border-radius: 4px;
        }
        QPushButton {
            background-color: #4b6eaf;
            color: white;
            border-radius: 6px;
            padding: 6px;
        }
        QPushButton:hover {
            background-color: #6a88c7;
        }
        QProgressBar {
            border: 1px solid #555;
            border-radius: 5px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #4b6eaf;
            width: 20px;
        }
        QLabel {
            font-weight: bold;
        }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
