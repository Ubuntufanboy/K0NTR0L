#!/usr/bin/env python3
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
from ctypes import wintypes

# --- WINDOWS LIBRARIES ---
import win32gui
import win32api
import win32con

# --- LIBRARIES ---
from pynput import keyboard, mouse
import mss
import numpy as np
from PIL import Image

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
#  NATIVE XINPUT WRAPPER (PASSIVE POLLING)
# ==========================================
# This allows us to read the controller without locking it from the emulator.
# Requires: Controller to be XInput compatible (Xbox controller or wrapped via x360ce)

class XInput:
    class XINPUT_BUTTONS(ctypes.Structure):
        _fields_ = [("wButtons", ctypes.c_ushort)]

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

    def __init__(self):
        # Try to load XInput 1.4 (Win 8+) or 1.3 (DirectX 9)
        try:
            self.xinput = ctypes.windll.xinput1_4
        except OSError:
            try:
                self.xinput = ctypes.windll.xinput1_3
            except OSError:
                self.xinput = None
        
        self.connected = False

    def get_state(self, index=0):
        if not self.xinput:
            return None

        state = self.XINPUT_STATE()
        # 0 = Success, 1167 = Device Not Connected
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
            # 1. Stitch Images
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
        # CSS FIX HERE
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
        self.server_url = "https://incorporate-jpg-nutten-offered.trycloudflare.com/upload"

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

        # XInput Setup
        self.xinput = XInput()
        
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
        self.kb_listener = keyboard.Listener(on_press=self.on_kb_press, on_release=self.on_kb_release)
        self.kb_listener.start()

        self.mouse_listener = mouse.Listener(on_click=self.on_mouse_click)
        self.mouse_listener.start()

        self.screenshot_timer = QTimer()
        self.screenshot_timer.timeout.connect(self.capture_frame)
        self.screenshot_timer.start(125) 

        self.input_event_signal.connect(self.handle_input_event)

    # --- INPUT HANDLERS ---
    
    def is_target_window_active(self):
        return win32gui.GetForegroundWindow() == self.selected_window['id']

    def on_kb_press(self, key):
        try:
            if self.is_target_window_active():
                k = key.char if hasattr(key, 'char') else str(key)
                self.input_event_signal.emit(f"Key_{k}", True)
        except: pass

    def on_kb_release(self, key):
        try:
            k = key.char if hasattr(key, 'char') else str(key)
            self.input_event_signal.emit(f"Key_{k}", False)
        except: pass

    def on_mouse_click(self, x, y, button, pressed):
        try:
            # Raycasting Check: Only record if mouse is ACTUALLY over our window
            hwnd_under_mouse = win32gui.WindowFromPoint((x, y))
            
            is_correct_window = False
            if hwnd_under_mouse == self.selected_window['id']:
                is_correct_window = True
            else:
                # Check parent hierarchy (handles UI elements inside window)
                parent = hwnd_under_mouse
                while parent:
                    parent = win32gui.GetParent(parent)
                    if parent == self.selected_window['id']:
                        is_correct_window = True
                        break

            if is_correct_window:
                btn_str = str(button).replace('Button.', 'Mouse')
                self.input_event_signal.emit(btn_str, pressed)
        except Exception:
            pass

    def handle_input_event(self, key_name, is_press):
        if is_press:
            self.active_keys.add(key_name)
            self.last_input_time = time.time()
            if not self.is_collecting:
                self.is_collecting = True
                self.status_label.setText("Status: Collecting")
        else:
            if key_name in self.active_keys:
                self.active_keys.remove(key_name)

    def capture_frame(self):
        if self.is_collecting and (time.time() - self.last_input_time > self.idle_threshold):
            self.is_collecting = False
            self.status_label.setText("Status: Idle")
            self.active_keys.clear()

        if not self.is_collecting:
            # Poll controller just to see if it starts activity
            gp = self.xinput.get_state(0)
            if gp and gp.wButtons > 0:
                 self.is_collecting = True
                 self.status_label.setText("Status: Collecting")
            else:
                return

        try:
            # 1. Controller Polling (XInput Snooping)
            # This does not lock the device. It just reads memory.
            gamepad = self.xinput.get_state(0)
            
            # Update GUI label once
            if self.xinput.connected:
                self.info_label.setText("Controller: Connected")
            else:
                self.info_label.setText("Controller: Not Found (Try x360ce)")

            # Process Buttons
            if gamepad:
                # Map XInput buttons to strings
                # A=4096, B=8192, X=16384, Y=32768, etc.
                btns = gamepad.wButtons
                button_map = {
                    0x1000: 'Btn_A', 0x2000: 'Btn_B', 0x4000: 'Btn_X', 0x8000: 'Btn_Y',
                    0x0001: 'DPad_Up', 0x0002: 'DPad_Down', 0x0004: 'DPad_Left', 0x0008: 'DPad_Right',
                    0x0010: 'Start', 0x0020: 'Back', 0x0100: 'LB', 0x0200: 'RB'
                }
                
                found_activity = False
                for mask, name in button_map.items():
                    if btns & mask:
                        self.active_keys.add(name)
                        found_activity = True
                    elif name in self.active_keys:
                        self.active_keys.remove(name)
                
                if found_activity or abs(gamepad.sThumbLX) > 2000 or abs(gamepad.bRightTrigger) > 10:
                    self.last_input_time = time.time()

            # 2. Capture Screen
            rect = win32gui.GetWindowRect(self.selected_window['id'])
            x, y = int(rect[0]), int(rect[1])
            w, h = int(rect[2]-rect[0]), int(rect[3]-rect[1])
            
            if w <= 0 or h <= 0: return

            with mss.mss() as sct:
                monitor = {"top": y, "left": x, "width": w, "height": h}
                shot = sct.grab(monitor)
                img = Image.frombytes("RGB", shot.size, shot.rgb).resize((256, 256), Image.LANCZOS)

            # 3. Analog Data
            analog_parts = []
            
            # Mouse Relative
            mx, my = win32api.GetCursorPos()
            if self.is_target_window_active():
                analog_parts.append(f"MX:{mx - x}")
                analog_parts.append(f"MY:{my - y}")

            # Controller Axes (Normalized -1.0 to 1.0)
            if gamepad:
                def norm_axis(val):
                    return val / 32768.0
                
                # Deadzone filter (~10%)
                if abs(gamepad.sThumbLX) > 3000: analog_parts.append(f"LX:{norm_axis(gamepad.sThumbLX):.2f}")
                if abs(gamepad.sThumbLY) > 3000: analog_parts.append(f"LY:{norm_axis(gamepad.sThumbLY):.2f}")
                if abs(gamepad.sThumbRX) > 3000: analog_parts.append(f"RX:{norm_axis(gamepad.sThumbRX):.2f}")
                if abs(gamepad.sThumbRY) > 3000: analog_parts.append(f"RY:{norm_axis(gamepad.sThumbRY):.2f}")
                if gamepad.bLeftTrigger > 10: analog_parts.append(f"LT:{gamepad.bLeftTrigger/255.0:.2f}")
                if gamepad.bRightTrigger > 10: analog_parts.append(f"RT:{gamepad.bRightTrigger/255.0:.2f}")

            keys_str = "+".join(sorted(self.active_keys)) if self.active_keys else "None"
            analog_str = ";".join(analog_parts)

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
            print(f"Loop Error: {e}")

    def trigger_upload(self):
        frames = list(self.frames_buffer)
        dataset = list(self.data_buffer)
        
        self.frames_buffer.clear()
        self.data_buffer.clear()
        self.buffer_label.setText(f"Buffer: 0/{self.batch_size}")

        worker = UploadThread(frames, dataset, self.api_key, self.server_url)
        worker.upload_finished.connect(self.on_upload_finished)
        self.active_uploads.append(worker)
        worker.finished.connect(lambda: self.active_uploads.remove(worker) if worker in self.active_uploads else None)
        worker.start()

    def on_upload_finished(self, success, msg):
        if success:
            self.batch_count += 1
            print("Upload OK")
        else:
            print(f"Upload Fail: {msg}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    c = DataCollector()
    c.show()
    sys.exit(app.exec_())
