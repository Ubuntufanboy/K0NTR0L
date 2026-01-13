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

# --- INPUT LIBRARY ---
import pygame 

# --- WINDOWS LIBRARIES ---
import win32gui
import win32api

# --- SCREEN CAPTURE ---
from pynput import keyboard, mouse
import mss
import numpy as np
from PIL import Image

# --- GUI LIBRARIES ---
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QDialog, QListWidget, QCheckBox, QMessageBox)
from PyQt5.QtCore import QTimer, pyqtSignal, QThread, Qt
from PyQt5.QtGui import QPixmap, QImage

# --- HIGH DPI FIX ---
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# ==========================================
#  CONFIGURATION
# ==========================================

SERVER_URL = "https://neurosama.jiemonlabs.help/upload"

# Noise Gates (Filtering)
DEADZONE_STICK = 0.08      
IDLE_THRESHOLD = 5.0       

# ==========================================
#  WORKER THREAD (UPLOAD)
# ==========================================

class UploadThread(QThread):
    upload_finished = pyqtSignal(bool, str)

    # FIX 1: Added server_url to arguments to match the call site
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

            # 3. Upload
            cmd = [
                'curl', '-X', 'POST',
                '-H', f'X-API-KEY: {self.api_key}',
                '-F', f'dataset=@{csv_tmp_path}',
                '-F', f'images=@{img_tmp_path}',
                self.server_url
            ]
            
            # Using encoding='utf-8' to prevent issues on Windows
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
                    try: os.unlink(p)
                    except OSError: pass

# ==========================================
#  GUI CLASSES
# ==========================================

class APIKeyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("API Key Required")
        self.setStyleSheet("background-color: #000; color: #FFF;")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Enter API Key:"))
        self.api_input = QLineEdit()
        self.api_input.setStyleSheet("background-color: #222; color: white; border: 1px solid #444;")
        layout.addWidget(self.api_input)
        btn = QPushButton("OK")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)
        self.setLayout(layout)

    def get_api_key(self):
        return self.api_input.text().strip()

class PreviewDialog(QDialog):
    def __init__(self, pil_image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preview Capture")
        self.setStyleSheet("background-color: #111; color: white;")
        layout = QVBoxLayout()
        
        # Convert PIL Image to QPixmap
        if pil_image:
            data = pil_image.tobytes("raw", "RGB")
            qim = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qim)
            
            lbl = QLabel()
            lbl.setPixmap(pix)
            lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(lbl)
            layout.addWidget(QLabel(f"Resolution: {pil_image.width}x{pil_image.height}"))
        else:
            layout.addWidget(QLabel("Could not capture image."))

        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)
        self.setLayout(layout)

class WindowSelector(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Window")
        self.setStyleSheet("background-color: #000; color: #FFF;")
        self.selected_window = None
        
        layout = QVBoxLayout()
        self.window_list = QListWidget()
        self.window_list.setStyleSheet("background-color: #222; color: white; border: 1px solid #444;")
        layout.addWidget(self.window_list)
        
        refresh = QPushButton("Refresh List")
        refresh.clicked.connect(self.populate_windows)
        layout.addWidget(refresh)

        self.ok_btn = QPushButton("Select & Start Recording")
        self.ok_btn.setStyleSheet("background-color: #006600; color: white; padding: 10px;")
        self.ok_btn.clicked.connect(self.accept_selection)
        layout.addWidget(self.ok_btn)
        
        self.setLayout(layout)
        self.populate_windows()

    def populate_windows(self):
        self.window_list.clear()
        def enum_cb(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
                windows.append((hwnd, win32gui.GetWindowText(hwnd)))
        windows = []
        win32gui.EnumWindows(enum_cb, windows)
        for hwnd, title in sorted(windows, key=lambda x: x[1].lower()):
            self.window_list.addItem(f"{title} (ID: {hwnd})")

    def accept_selection(self):
        item = self.window_list.currentItem()
        if item:
            try:
                wid = int(item.text().split("ID: ")[1].rstrip(")"))
                self.selected_window = {'id': wid}
                self.accept()
            except: pass

# ==========================================
#  MAIN COLLECTOR
# ==========================================

class DataCollector(QMainWindow):
    input_event_signal = pyqtSignal(str, bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuro Passive Data Collector")
        self.config_path = Path.home() / ".neuro_collector_config.json"
        
        self.api_key = self.load_api_key() or self.prompt_api_key()
        if not self.api_key: sys.exit(0)

        # Buffer State
        self.selected_window = None
        self.frames_buffer = []
        self.data_buffer = [] 
        self.active_keys = set()
        
        # Collection State
        self.frame_count = 0
        self.batch_count = 0
        self.batch_size = 196
        self.is_collecting = False
        self.last_input_time = 0
        self.active_uploads = []
        
        # Controller State
        self.joystick = None
        self.prev_input_hash = "" 
        
        pygame.init()
        pygame.joystick.init()
        
        self.init_ui()
        self.select_window()
        
        if self.selected_window:
            self.setup_collectors()
        else:
            sys.exit(0)

    def init_ui(self):
        self.setStyleSheet("background-color: #111; color: white;")
        self.setFixedSize(450, 250) # Increased height for new button
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        
        self.status_label = QLabel("Status: Waiting")
        self.status_label.setStyleSheet("color: #00FF00; font-weight: bold; font-size: 16px;")
        layout.addWidget(self.status_label)
        
        self.info_label = QLabel(f"Controller: Searching...")
        self.info_label.setStyleSheet("color: #AAAAAA;")
        layout.addWidget(self.info_label)

        self.debug_chk = QCheckBox("Show Raw Input Values")
        self.debug_chk.setStyleSheet("color: #FFCC00;")
        layout.addWidget(self.debug_chk)

        # FIX 2: Added Preview Button
        self.preview_btn = QPushButton("Preview Capture Region")
        self.preview_btn.setStyleSheet("background-color: #333; color: cyan; border: 1px solid cyan;")
        self.preview_btn.clicked.connect(self.show_preview)
        layout.addWidget(self.preview_btn)

        self.stats_label = QLabel("Frames: 0 | Batches: 0")
        layout.addWidget(self.stats_label)

        self.buffer_label = QLabel("Buffer: 0/196")
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

    def show_preview(self):
        if not self.selected_window:
            return
        
        img = self.capture_screen()
        if img:
            d = PreviewDialog(img, self)
            d.exec_()
        else:
            QMessageBox.warning(self, "Error", "Could not grab window. Is it minimized?")

    def capture_screen(self):
        """Helper to capture the screen, used by both record and preview."""
        try:
            rect = win32gui.GetWindowRect(self.selected_window['id'])
            x, y = int(rect[0]), int(rect[1])
            w, h = int(rect[2]-rect[0]), int(rect[3]-rect[1])
            if w <= 0 or h <= 0: return None

            with mss.mss() as sct:
                monitor = {"top": y, "left": x, "width": w, "height": h}
                shot = sct.grab(monitor)
                img = Image.frombytes("RGB", shot.size, shot.rgb).resize((256, 256), Image.LANCZOS)
                return img
        except Exception as e:
            print(f"Capture error: {e}")
            return None

    def setup_collectors(self):
        self.kb_listener = keyboard.Listener(on_press=self.on_kb_press, on_release=self.on_kb_release)
        self.kb_listener.start()
        self.mouse_listener = mouse.Listener(on_click=self.on_mouse_click)
        self.mouse_listener.start()
        
        self.screenshot_timer = QTimer()
        self.screenshot_timer.timeout.connect(self.loop_tick)
        self.screenshot_timer.start(100)
        
        self.input_event_signal.connect(self.handle_input_event)

    def refresh_controllers(self):
        pygame.event.pump()
        if pygame.joystick.get_count() > 0:
            if self.joystick is None:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                self.info_label.setText(f"Controller: {self.joystick.get_name()} (Passive)")
        else:
            self.joystick = None
            self.info_label.setText("Controller: Disconnected")

    def loop_tick(self):
        self.refresh_controllers()
        
        active_input = False
        raw_analog_data = []
        raw_buttons_data = []
        
        if self.joystick:
            pygame.event.pump()
            
            # --- READ RAW AXES ---
            axes_count = self.joystick.get_numaxes()
            for i in range(axes_count):
                val = self.joystick.get_axis(i)
                if abs(val) > DEADZONE_STICK:
                    raw_analog_data.append(f"Ax{i}:{val:.3f}")
                    active_input = True

            # --- READ RAW BUTTONS ---
            btn_count = self.joystick.get_numbuttons()
            for i in range(btn_count):
                if self.joystick.get_button(i):
                    raw_buttons_data.append(f"Btn{i}")
                    active_input = True
            
            # --- FIX 3: READ HATS (D-PAD / C-BUTTONS) ---
            # N64/Retro adapters often map C-buttons or D-pads to 'Hats'
            hat_count = self.joystick.get_numhats()
            for i in range(hat_count):
                # Hat returns a tuple (x, y) where x/y are -1, 0, or 1
                hat_val = self.joystick.get_hat(i)
                if hat_val != (0, 0):
                    raw_buttons_data.append(f"Hat{i}:{hat_val}")
                    active_input = True
            
            # --- DEBUG LOG ---
            if self.debug_chk.isChecked() and active_input:
                print(f"[RAW] Buttons/Hats: {raw_buttons_data} | Axes: {raw_analog_data}")

        # --- IDLE LOGIC ---
        if active_input or len(self.active_keys) > 0:
            self.last_input_time = time.time()
            if not self.is_collecting:
                self.is_collecting = True
                self.status_label.setText("Status: Collecting")
        
        if self.is_collecting and (time.time() - self.last_input_time > IDLE_THRESHOLD):
            self.is_collecting = False
            self.status_label.setText("Status: Idle")
            self.active_keys.clear()
        
        # --- CAPTURE ---
        if self.is_collecting:
            self.record_frame(raw_buttons_data, raw_analog_data)

    def record_frame(self, buttons, analog):
        try:
            # Replaced manual mss code with helper function
            img = self.capture_screen()
            if img is None: return

            all_keys = list(buttons)
            if self.active_keys:
                all_keys.extend(self.active_keys)
            
            keys_str = "+".join(all_keys) if all_keys else "None"
            analog_str = ";".join(analog)

            self.frames_buffer.append(np.array(img))
            self.data_buffer.append({
                'timestamp': time.time(),
                'frame_index': len(self.frames_buffer) - 1,
                'keys': keys_str,
                'analog': analog_str
            })
            
            self.frame_count += 1
            self.stats_label.setText(f"Frames: {self.frame_count} | Batches: {self.batch_count}")
            self.buffer_label.setText(f"Buffer: {len(self.frames_buffer)}/{self.batch_size}")

            if len(self.frames_buffer) >= self.batch_size:
                self.trigger_upload()
        except Exception as e:
            print(f"Capture error: {e}")

    def trigger_upload(self):
        frames = list(self.frames_buffer)
        dataset = list(self.data_buffer)
        self.frames_buffer.clear()
        self.data_buffer.clear()
        self.buffer_label.setText(f"Buffer: 0/{self.batch_size}")
        
        # Fixed arguments in call
        worker = UploadThread(frames, dataset, self.api_key, SERVER_URL)
        worker.upload_finished.connect(self.on_upload_finished)
        self.active_uploads.append(worker)
        worker.start()

    def on_upload_finished(self, success, msg):
        if success:
            self.batch_count += 1
            print("Upload OK")
        else:
            print(f"Upload Fail: {msg}")

    # --- KEYBOARD/MOUSE LISTENERS ---
    def on_kb_press(self, key):
        if win32gui.GetForegroundWindow() == self.selected_window['id']:
            k = str(key).replace("'", "")
            self.input_event_signal.emit(f"K_{k}", True)

    def on_kb_release(self, key):
        k = str(key).replace("'", "")
        self.input_event_signal.emit(f"K_{k}", False)

    def on_mouse_click(self, x, y, button, pressed):
        if win32gui.GetForegroundWindow() == self.selected_window['id']:
            self.input_event_signal.emit(f"M_{button}", pressed)

    def handle_input_event(self, key, pressed):
        if pressed: self.active_keys.add(key)
        elif key in self.active_keys: self.active_keys.remove(key)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    c = DataCollector()
    c.show()
    sys.exit(app.exec_())
