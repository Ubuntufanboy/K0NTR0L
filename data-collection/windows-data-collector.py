import sys
import time
import csv
import io
import json
import os
import subprocess
import tempfile
import ctypes
from pathlib import Path

# --- WINDOWS LIBRARIES ---
import win32gui
import win32api
import win32con

# --- INPUT LIBRARIES ---
from pynput import keyboard, mouse
import mss
import numpy as np
from PIL import Image

# --- GUI LIBRARIES ---
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QDialog, QListWidget, QMessageBox)
from PyQt5.QtCore import QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap

# --- HIGH DPI FIX ---
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# ==========================================
#  XINPUT STRUCTURES (GLOBAL SCOPE)
# ==========================================

class XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [("wButtons", ctypes.c_ushort),
                ("bLeftTrigger", ctypes.c_ubyte),
                ("bRightTrigger", ctypes.c_ubyte),
                ("sThumbLX", ctypes.c_short),
                ("sThumbLY", ctypes.c_short),
                ("sThumbRX", ctypes.c_short),
                ("sThumbRY", ctypes.c_short)]

class XINPUT_STATE(ctypes.Structure):
    _fields_ = [("dwPacketNumber", ctypes.c_ulong),
                ("Gamepad", XINPUT_GAMEPAD)]

class XInputWrapper:
    def __init__(self):
        # Try to load XInput 1.4 (Windows 8+) or 1.3 (DirectX 9/Win7)
        self.xinput = None
        self.connected = False
        
        try:
            self.xinput = ctypes.windll.xinput1_4
        except OSError:
            try:
                self.xinput = ctypes.windll.xinput1_3
            except OSError:
                self.xinput = None
        
    def get_state(self, index=0):
        if not self.xinput:
            return None

        state = XINPUT_STATE()
        # XInputGetState returns 0 (ERROR_SUCCESS) if connected
        res = self.xinput.XInputGetState(index, ctypes.byref(state))
        
        if res == 0:
            self.connected = True
            return state.Gamepad
        else:
            self.connected = False
            return None

# ==========================================
#  WORKER THREAD
# ==========================================

class UploadThread(QThread):
    upload_finished = pyqtSignal(bool, str)

    def __init__(self, frames, dataset, api_key, server_url):
        super().__init__()
        self.frames = frames
        self.dataset = dataset
        self.api_key = api_key
        self.server_url = server_url

    def run(self):
        img_tmp_path = None
        csv_tmp_path = None
        
        try:
            # 1. Stitch Images (14x14 grid)
            grid_size = 14
            img_size = 256
            total_size = grid_size * img_size
            big_img = Image.new('RGB', (total_size, total_size), (0, 0, 0))

            for idx, frame in enumerate(self.frames):
                row = idx // grid_size
                col = idx % grid_size
                big_img.paste(Image.fromarray(frame), (col * img_size, row * img_size))

            with tempfile.NamedTemporaryFile(suffix='.webp', delete=False) as img_tmp:
                big_img.save(img_tmp, format='WEBP', quality=70, method=6)
                img_tmp_path = img_tmp.name

            # 2. Save CSV
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8', newline='') as csv_tmp:
                fieldnames = ['timestamp', 'frame_index', 'keys', 'analog']
                writer = csv.DictWriter(csv_tmp, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.dataset)
                csv_tmp_path = csv_tmp.name

            # 3. Upload via Curl
            cmd = [
                'curl', '-X', 'POST',
                '-H', f'X-API-KEY: {self.api_key}',
                '-F', f'dataset=@{csv_tmp_path}',
                '-F', f'images=@{img_tmp_path}',
                self.server_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

            if result.returncode == 0:
                self.upload_finished.emit(True, result.stdout)
            else:
                self.upload_finished.emit(False, result.stderr)

        except Exception as e:
            self.upload_finished.emit(False, str(e))
            
        finally:
            for p in [img_tmp_path, csv_tmp_path]:
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

# ==========================================
#  GUI CLASSES
# ==========================================

class APIKeyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("API Key Required")
        self.setStyleSheet("background-color: #000000; color: #FFFFFF;")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Enter API Key:"))
        self.api_input = QLineEdit()
        self.api_input.setStyleSheet("background-color: #1a1a1a; color: white; border: 1px solid #333;")
        layout.addWidget(self.api_input)
        self.ok_btn = QPushButton("OK")
        self.ok_btn.setStyleSheet("background-color: #2a2a2a; color: white;")
        self.ok_btn.clicked.connect(self.accept)
        layout.addWidget(self.ok_btn)
        self.setLayout(layout)

    def get_api_key(self):
        return self.api_input.text().strip()

class WindowSelector(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Window")
        self.setStyleSheet("background-color: #000000; color: #FFFFFF;")
        self.selected_window = None
        
        layout = QVBoxLayout()
        self.window_list = QListWidget()
        self.window_list.setStyleSheet("background-color: #1a1a1a; color: white; border: 1px solid #333;")
        layout.addWidget(self.window_list)
        
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(400, 300)
        self.preview_label.setStyleSheet("border: 1px solid #333; background-color: #000;")
        self.preview_label.setScaledContents(True)
        layout.addWidget(self.preview_label)

        self.preview_btn = QPushButton("Preview")
        self.preview_btn.setStyleSheet("background-color: #2a2a2a; color: white;")
        self.preview_btn.clicked.connect(self.preview_window)
        
        self.ok_btn = QPushButton("OK")
        self.ok_btn.setStyleSheet("background-color: #2a2a2a; color: white;")
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setEnabled(False)
        
        layout.addWidget(self.preview_btn)
        layout.addWidget(self.ok_btn)
        self.setLayout(layout)
        self.populate_windows()

    def populate_windows(self):
        def enum_cb(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
                windows.append((hwnd, win32gui.GetWindowText(hwnd)))
        windows = []
        win32gui.EnumWindows(enum_cb, windows)
        for hwnd, title in sorted(windows, key=lambda x: x[1].lower()):
            self.window_list.addItem(f"{title} (ID: {hwnd})")

    def preview_window(self):
        selected = self.window_list.currentItem()
        if not selected: return
        
        try:
            wid = int(selected.text().split("ID: ")[1].rstrip(")"))
            rect = win32gui.GetWindowRect(wid)
            
            x, y = int(rect[0]), int(rect[1])
            w, h = int(rect[2] - rect[0]), int(rect[3] - rect[1])
            
            if w <= 0 or h <= 0:
                QMessageBox.warning(self, "Error", "Window is minimized or invalid.")
                return

            with mss.mss() as sct:
                monitor = {'top': y, 'left': x, 'width': w, 'height': h}
                shot = sct.grab(monitor)
                img = Image.frombytes("RGB", shot.size, shot.rgb)
                
                img.thumbnail((400, 300), Image.LANCZOS)
                data = io.BytesIO()
                img.save(data, format='PNG')
                pixmap = QPixmap()
                pixmap.loadFromData(data.getvalue())
                self.preview_label.setPixmap(pixmap)
                
                self.selected_window = {'id': wid, 'x': x, 'y': y, 'width': w, 'height': h}
                self.ok_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.warning(self, "Preview Error", f"Could not preview: {e}")

# ==========================================
#  MAIN COLLECTOR
# ==========================================

class DataCollector(QMainWindow):
    input_event_signal = pyqtSignal(str, bool) 

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuro Data Collector")
        self.config_path = Path.home() / ".neuro_collector_config.json"
        self.api_key = self.load_api_key() or self.prompt_api_key()
        if not self.api_key: sys.exit(0)
        self.server_url = "https://neurosama.jiemonlabs.help/upload"

        # State
        self.selected_window = None
        self.frames_buffer = []
        self.data_buffer = [] 
        self.active_keys = set()
        
        self.frame_count = 0
        self.batch_count = 0
        self.batch_size = 196
        self.is_collecting = False
        self.last_input_time = 0
        self.idle_threshold = 5.0
        self.active_uploads = []

        # Initialize XInput Wrapper
        self.xinput = XInputWrapper()
        
        self.init_ui()
        self.select_window()
        
        if self.selected_window:
            self.setup_collectors()
        else:
            sys.exit(0)

    def init_ui(self):
        self.setStyleSheet("background-color: #000000;")
        self.setFixedSize(400, 220)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        
        self.status_label = QLabel("Status: Waiting")
        self.status_label.setStyleSheet("color: #00FF00; font-weight: bold; font-size: 14px;")
        layout.addWidget(self.status_label)
        
        self.info_label = QLabel(f"Controller: Searching (XInput)...")
        self.info_label.setStyleSheet("color: #AAAAAA;")
        layout.addWidget(self.info_label)

        self.stats_label = QLabel("Frames: 0 | Batches: 0")
        self.stats_label.setStyleSheet("color: white;")
        layout.addWidget(self.stats_label)

        self.buffer_label = QLabel("Buffer: 0/196")
        self.buffer_label.setStyleSheet("color: white;")
        layout.addWidget(self.buffer_label)
        central.setLayout(layout)

    def load_api_key(self):
        if self.config_path.exists():
            try: return json.load(open(self.config_path))['api_key']
            except: pass
        return None

    def prompt_api_key(self):
        d = APIKeyDialog(self)
        if d.exec_() == QDialog.Accepted:
            k = d.get_api_key()
            json.dump({'api_key': k}, open(self.config_path, 'w'))
            return k
        return None

    def select_window(self):
        d = WindowSelector(self)
        if d.exec_() == QDialog.Accepted:
            self.selected_window = d.selected_window

    def setup_collectors(self):
        self.kb_listener = keyboard.Listener(on_press=self.on_kb_press, on_release=self.on_kb_
