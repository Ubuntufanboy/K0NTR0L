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
from collections import deque
from datetime import datetime

import win32gui
import win32con
import win32api

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QDialog, QListWidget, QMessageBox)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QPalette, QColor, QPixmap
import numpy as np
from PIL import Image
from pynput import keyboard
import mss

# This ensures coordinate calculations match what MSS sees on the screen
# Windows is weird about DPI so this is required
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

class UploadThread(QThread):
    upload_finished = pyqtSignal(bool, str)

    def __init__(self, frames, keypresses, api_key, server_url):
        super().__init__()
        self.frames = frames
        self.keypresses = keypresses
        self.api_key = api_key
        self.server_url = server_url

    def run(self):
        img_tmp_path = None
        csv_tmp_path = None
        
        try:            
            # Create 3584x3584 image (14x14 grid of 256x256 images)
            grid_size = 14
            img_size = 256
            total_size = grid_size * img_size

            big_img = Image.new('RGB', (total_size, total_size), (0, 0, 0))

            for idx, frame in enumerate(self.frames):
                row = idx // grid_size
                col = idx % grid_size
                x = col * img_size
                y = row * img_size
                frame_img = Image.fromarray(frame)
                big_img.paste(frame_img, (x, y))

            with tempfile.NamedTemporaryFile(suffix='.webp', delete=False) as img_tmp: # Lesson learned... delete MUST be False
                big_img.save(img_tmp, format='WEBP', quality=70, method=6)
                img_tmp_path = img_tmp.name

            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8', newline='') as csv_tmp:
                writer = csv.DictWriter(csv_tmp, fieldnames=['timestamp', 'frame_index', 'keys'])
                writer.writeheader()
                writer.writerows(self.keypresses)
                csv_tmp_path = csv_tmp.name

            # MUST HAVE CURL
            cmd = [
                'curl',
                '-X', 'POST',
                '-H', f'X-API-KEY: {self.api_key}',
                '-F', f'dataset=@{csv_tmp_path}',
                '-F', f'images=@{img_tmp_path}',
                self.server_url
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                self.upload_finished.emit(True, result.stdout)
            else:
                self.upload_finished.emit(False, result.stderr)

        except Exception as e:
            self.upload_finished.emit(False, str(e))
            
        finally:
            if img_tmp_path and os.path.exists(img_tmp_path):
                try:
                    os.unlink(img_tmp_path)
                except OSError:
                    pass
            if csv_tmp_path and os.path.exists(csv_tmp_path):
                try:
                    os.unlink(csv_tmp_path)
                except OSError:
                    pass

class APIKeyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("API Key Required")
        self.setModal(True)
        self.setStyleSheet("background-color: #000000; color: #FFFFFF;")

        layout = QVBoxLayout()

        label = QLabel("Enter API Key:")
        label.setStyleSheet("color: #FFFFFF; font-size: 14px;")
        layout.addWidget(label)

        self.api_input = QLineEdit()
        self.api_input.setStyleSheet("""
            QLineEdit {
                background-color: #1a1a1a;
                color: #FFFFFF;
                border: 1px solid #333333;
                padding: 5px;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.api_input)

        btn_layout = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: #FFFFFF;
                border: 1px solid #444444;
                padding: 5px 15px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
        """)
        self.ok_btn.clicked.connect(self.accept)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet(self.ok_btn.styleSheet())
        self.cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def get_api_key(self):
        return self.api_input.text().strip()

class WindowSelector(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Window")
        self.setModal(True)
        self.setStyleSheet("background-color: #000000; color: #FFFFFF;")
        self.selected_window = None

        layout = QVBoxLayout()

        label = QLabel("Select a window to capture:")
        label.setStyleSheet("color: #FFFFFF; font-size: 14px;")
        layout.addWidget(label)

        self.window_list = QListWidget()
        self.window_list.setStyleSheet("""
            QListWidget {
                background-color: #1a1a1a;
                color: #FFFFFF;
                border: 1px solid #333333;
                font-size: 12px;
            }
            QListWidget::item:selected {
                background-color: #3a3a3a;
            }
        """)
        layout.addWidget(self.window_list)

        self.preview_label = QLabel()
        self.preview_label.setStyleSheet("border: 1px solid #333333;")
        self.preview_label.setFixedSize(400, 300)
        self.preview_label.setScaledContents(True)
        layout.addWidget(self.preview_label)

        btn_layout = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: #FFFFFF;
                border: 1px solid #444444;
                padding: 5px 15px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
        """)
        self.preview_btn.clicked.connect(self.preview_window)

        self.ok_btn = QPushButton("OK")
        self.ok_btn.setStyleSheet(self.preview_btn.styleSheet())
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setEnabled(False)

        btn_layout.addWidget(self.preview_btn)
        btn_layout.addWidget(self.ok_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.populate_windows()

    def populate_windows(self):
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
                windows.append((hwnd, win32gui.GetWindowText(hwnd)))
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        windows.sort(key=lambda x: x[1].lower())

        for hwnd, title in windows:
            item_text = f"{title} (ID: {hwnd})"
            self.window_list.addItem(item_text)

    def preview_window(self):
        selected = self.window_list.currentItem()
        if not selected:
            return

        text = selected.text()
        try:
            window_id = int(text.split("ID: ")[1].rstrip(")"))
        except IndexError:
            return

        try:
            rect = win32gui.GetWindowRect(window_id)
            x = rect[0]
            y = rect[1]
            w = rect[2] - x
            h = rect[3] - y

            # Don't allow minimized windows
            if w <= 0 or h <= 0:
                raise ValueError("Window has invalid dimensions (minimized?)")

            with mss.mss() as sct:
                monitor = {
                    "top": y,
                    "left": x,
                    "width": w,
                    "height": h
                }
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

                # Convert to QPixmap for preview
                img.thumbnail((400, 300), Image.LANCZOS)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')

                pixmap = QPixmap()
                pixmap.loadFromData(img_byte_arr.getvalue())
                self.preview_label.setPixmap(pixmap)

                self.selected_window = {
                    'id': window_id,
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h
                }
                self.ok_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to preview window: {str(e)}\nMake sure it is not minimized.")

class DataCollector(QMainWindow):
    key_press_signal = pyqtSignal(str)
    key_release_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuro Data Collector")
        self.config_path = Path.home() / ".neuro_collector_config.json"
        self.api_key = self.load_api_key()
        self.server_url = "https://incorporate-jpg-nutten-offered.trycloudflare.com/upload"

        if not self.api_key:
            self.api_key = self.prompt_api_key()
            if not self.api_key:
                sys.exit(0)

        self.selected_window = None
        self.frames_buffer = []
        self.keypresses_buffer = []
        self.active_keys = set()
        
        self.frame_count = 0
        self.batch_count = 0
        self.batch_size = 196
        self.is_collecting = False
        self.last_keypress_time = 0
        self.idle_threshold = 5.0
        
        self.active_uploads = []

        self.init_ui()
        self.select_window()

        if self.selected_window:
            self.setup_collectors()
        else:
            sys.exit(0)

    def init_ui(self):
        self.setStyleSheet("background-color: #000000;")
        self.setFixedSize(400, 200)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()

        self.status_label = QLabel("Status: Waiting")
        self.status_label.setStyleSheet("color: #00FF00; font-size: 16px; font-weight: bold;")
        layout.addWidget(self.status_label)

        self.frame_label = QLabel("Frames Captured: 0")
        self.frame_label.setStyleSheet("color: #FFFFFF; font-size: 14px;")
        layout.addWidget(self.frame_label)

        self.batch_label = QLabel("Batches Uploaded: 0")
        self.batch_label.setStyleSheet("color: #FFFFFF; font-size: 14px;")
        layout.addWidget(self.batch_label)

        self.buffer_label = QLabel("Buffer: 0/196")
        self.buffer_label.setStyleSheet("color: #FFFFFF; font-size: 14px;")
        layout.addWidget(self.buffer_label)

        central.setLayout(layout)

    def load_api_key(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    return config.get('api_key')
            except:
                pass
        return None

    def save_api_key(self, api_key):
        with open(self.config_path, 'w') as f:
            json.dump({'api_key': api_key}, f)

    def prompt_api_key(self):
        dialog = APIKeyDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            api_key = dialog.get_api_key()
            if api_key:
                self.save_api_key(api_key)
                return api_key
        return None

    def select_window(self):
        dialog = WindowSelector(self)
        if dialog.exec_() == QDialog.Accepted:
            self.selected_window = dialog.selected_window

    def setup_collectors(self):
        self.key_press_signal.connect(self.handle_key_press)
        self.key_release_signal.connect(self.handle_key_release)

        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_pynput_press,
            on_release=self.on_pynput_release
        )
        self.keyboard_listener.start()

        self.screenshot_timer = QTimer()
        self.screenshot_timer.timeout.connect(self.capture_frame)
        self.screenshot_timer.start(125)

        self.idle_timer = QTimer()
        self.idle_timer.timeout.connect(self.check_idle)
        self.idle_timer.start(100)

    def on_pynput_press(self, key):
        try:
            # I loveee the windows api 
            focused_hwnd = win32gui.GetForegroundWindow()
            
            if focused_hwnd == self.selected_window['id']:
                 key_str = key.char if hasattr(key, 'char') else str(key)
                 self.key_press_signal.emit(key_str)
        except:
            pass

    def on_pynput_release(self, key):
        try:
            key_str = key.char if hasattr(key, 'char') else str(key)
            self.key_release_signal.emit(key_str)
        except:
            pass

    def handle_key_press(self, key_str):
        self.active_keys.add(key_str)
        self.last_keypress_time = time.time()

        if not self.is_collecting:
            self.is_collecting = True
            self.status_label.setText("Status: Collecting")
            self.status_label.setStyleSheet("color: #00FF00; font-size: 16px; font-weight: bold;")

    def handle_key_release(self, key_str):
        # Safe remove if somebody alt tabs in basically
        if key_str in self.active_keys:
            self.active_keys.remove(key_str)

    def check_idle(self):
        if self.is_collecting:
            idle_time = time.time() - self.last_keypress_time
            if idle_time >= self.idle_threshold:
                self.is_collecting = False
                self.status_label.setText("Status: Idle")
                self.status_label.setStyleSheet("color: #FFAA00; font-size: 16px; font-weight: bold;")
                self.active_keys.clear() # Clear keys if we go AFK

    def capture_frame(self):
        if not self.is_collecting:
            return

        try:
            with mss.mss() as sct:
                try:
                    rect = win32gui.GetWindowRect(self.selected_window['id'])
                    x, y = rect[0], rect[1]
                    w, h = rect[2] - x, rect[3] - y
                except:
                    # If window was closed, stop collecting
                    self.is_collecting = False
                    self.status_label.setText("Status: Window Lost")
                    return

                monitor = {
                    "top": y,
                    "left": x,
                    "width": w,
                    "height": h
                }
                
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                img = img.resize((256, 256), Image.LANCZOS)

                timestamp = time.time()
                
                # Get string representation of currently held keys
                current_keys_str = "+".join(sorted(self.active_keys)) if self.active_keys else "None"

                self.frames_buffer.append(np.array(img))
                self.keypresses_buffer.append({
                    'timestamp': timestamp,
                    'frame_index': len(self.frames_buffer) - 1,
                    'keys': current_keys_str
                })

                self.frame_count += 1
                self.frame_label.setText(f"Frames Captured: {self.frame_count}")
                self.buffer_label.setText(f"Buffer: {len(self.frames_buffer)}/{self.batch_size}")

                if len(self.frames_buffer) >= self.batch_size:
                    self.trigger_upload()
        except Exception as e:
            print(f"Capture error: {e}")

    def trigger_upload(self):
        frames_to_upload = list(self.frames_buffer)
        keys_to_upload = list(self.keypresses_buffer)
        
        self.frames_buffer.clear()
        self.keypresses_buffer.clear()
        self.buffer_label.setText(f"Buffer: 0/{self.batch_size}")
        
        print(f"Starting background upload. Last keys recorded: {keys_to_upload[-1]['keys']}")

        worker = UploadThread(frames_to_upload, keys_to_upload, self.api_key, self.server_url)
        worker.upload_finished.connect(self.on_upload_finished)
        self.active_uploads.append(worker)
        worker.finished.connect(lambda: self.cleanup_thread(worker))
        worker.start()

    def on_upload_finished(self, success, message):
        if success:
            self.batch_count += 1
            self.batch_label.setText(f"Batches Uploaded: {self.batch_count}")
            print(f"Upload Success: {message}")
        else:
            print(f"Upload Failed: {message}")

    def cleanup_thread(self, worker):
        if worker in self.active_uploads:
            self.active_uploads.remove(worker)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    collector = DataCollector()
    collector.show()
    sys.exit(app.exec_())
