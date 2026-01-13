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
import win32con

# --- INPUT & SCREEN CAPTURE ---
from pynput import keyboard, mouse
import mss
import numpy as np
from PIL import Image

# --- VIRTUAL CONTROLLER ---
try:
    import vgamepad as vg
except ImportError:
    print("ERROR: Missing 'vgamepad'. Install it via: pip install vgamepad")
    sys.exit(1)

# --- GUI LIBRARIES ---
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QDialog, QListWidget, QCheckBox, QMessageBox, QProgressBar)
from PyQt5.QtCore import QTimer, pyqtSignal, QThread, Qt
from PyQt5.QtGui import QPixmap

# --- HIGH DPI FIX ---
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# ==========================================
#  CONFIGURATION
# ==========================================

SERVER_URL = "https://neurosama.jiemonlabs.help/upload"
DEADZONE_STICK = 0.15       # Analog value must be > 0.15 to register as input
IDLE_NOISE_GATE = 0.02      # Analog change must be > 0.02 to reset idle timer

# ==========================================
#  WORKER THREAD (UPLOAD)
# ==========================================

class UploadThread(QThread):
    upload_finished = pyqtSignal(bool, str)

    def __init__(self, frames, dataset, api_key):
        super().__init__()
        self.frames = frames
        self.dataset = dataset
        self.api_key = api_key

    def run(self):
        img_tmp_path = None
        csv_tmp_path = None
        try:
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

            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8', newline='') as csv_tmp:
                fieldnames = ['timestamp', 'frame_index', 'keys', 'analog']
                writer = csv.DictWriter(csv_tmp, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.dataset)
                csv_tmp_path = csv_tmp.name

            cmd = [
                'curl', '-X', 'POST',
                '-H', f'X-API-KEY: {self.api_key}',
                '-F', f'dataset=@{csv_tmp_path}',
                '-F', f'images=@{img_tmp_path}',
                SERVER_URL
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
                    try: os.unlink(p)
                    except OSError: pass

# ==========================================
#  HELPER DIALOGS
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

        btn_box = QHBoxLayout()
        ok_btn = QPushButton("Select")
        ok_btn.setStyleSheet("background-color: #006600; color: white;")
        ok_btn.clicked.connect(self.accept_selection)
        btn_box.addWidget(ok_btn)
        layout.addLayout(btn_box)
        
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
#  MAPPING DIALOG
# ==========================================

class ControllerMapper(QDialog):
    def __init__(self, joystick, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Map Controller Buttons")
        self.setStyleSheet("background-color: #111; color: white;")
        self.joystick = joystick
        self.resize(400, 250)
        self.mapping = {}
        
        # Mapping Sequence: Prompt -> Xbox Output Code
        self.targets = [
            ("A Button", vg.XUSB_BUTTON.XUSB_GAMEPAD_A),
            ("B Button", vg.XUSB_BUTTON.XUSB_GAMEPAD_B),
            ("Start Button", vg.XUSB_BUTTON.XUSB_GAMEPAD_START),
            ("Z Trigger (Maps to LB)", vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER),
            ("R Trigger (Maps to RB)", vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER),
            ("C-Up (Maps to Y)", vg.XUSB_BUTTON.XUSB_GAMEPAD_Y),
            ("C-Right (Maps to X)", vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        ]
        self.current_idx = 0
        
        layout = QVBoxLayout()
        self.lbl_instruction = QLabel(f"PRESS BUTTON FOR:\n\n{self.targets[0][0]}")
        self.lbl_instruction.setAlignment(Qt.AlignCenter)
        self.lbl_instruction.setStyleSheet("font-size: 20px; font-weight: bold; color: yellow;")
        layout.addWidget(self.lbl_instruction)
        
        self.progress = QProgressBar()
        self.progress.setRange(0, len(self.targets))
        layout.addWidget(self.progress)
        
        self.setLayout(layout)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.poll_input)
        self.timer.start(50)
        
        self.last_buttons = [False] * 32 # Buffer for up to 32 buttons

    def poll_input(self):
        pygame.event.pump()
        num_buttons = self.joystick.get_numbuttons()
        
        for i in range(num_buttons):
            pressed = self.joystick.get_button(i)
            # Detect Rising Edge (Press down)
            if pressed and not self.last_buttons[i]:
                self.map_button(i)
                break # Only accept one button at a time
            
            if i < len(self.last_buttons):
                self.last_buttons[i] = pressed

    def map_button(self, physical_id):
        target_name, xbox_code = self.targets[self.current_idx]
        self.mapping[physical_id] = xbox_code
        print(f"Mapped Physical ID {physical_id} -> {target_name}")
        
        self.current_idx += 1
        self.progress.setValue(self.current_idx)
        
        if self.current_idx >= len(self.targets):
            self.accept()
        else:
            self.lbl_instruction.setText(f"PRESS BUTTON FOR:\n\n{self.targets[self.current_idx][0]}")
            self.last_buttons = [False] * 32 # Clear buffer to prevent double mapping

# ==========================================
#  MAIN COLLECTOR
# ==========================================

class DataCollector(QMainWindow):
    input_event_signal = pyqtSignal(str, bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuro Data Collector v2.1")
        self.config_path = Path.home() / ".neuro_collector_config.json"
        self.map_path = Path.home() / ".neuro_controller_map.json"
        
        self.api_key = self.load_api_key() or self.prompt_api_key()
        if not self.api_key: sys.exit(0)

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
        
        # Idle Logic State
        self.prev_axes = {} 
        
        # Controller State
        self.joystick = None
        self.button_map = {} # Physical ID -> Xbox Code
        pygame.init()
        pygame.joystick.init()
        
        try:
            self.virtual_gamepad = vg.VX360Gamepad()
            print("Virtual Controller Initialized.")
        except Exception as e:
            print(f"Failed to init virtual controller: {e}")
            sys.exit(1)

        self.init_ui()
        self.load_mapping()
        
        # Start Window Selection
        self.select_window()
        if self.selected_window:
            self.setup_collectors()
        else:
            sys.exit(0)

    def init_ui(self):
        self.setStyleSheet("background-color: #111; color: white;")
        self.setFixedSize(500, 320)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        
        self.status_label = QLabel("Status: Waiting")
        self.status_label.setStyleSheet("color: #00FF00; font-weight: bold; font-size: 16px;")
        layout.addWidget(self.status_label)
        
        self.info_label = QLabel(f"Controller: Searching...")
        self.info_label.setStyleSheet("color: #AAAAAA;")
        layout.addWidget(self.info_label)

        # Map Button
        self.map_btn = QPushButton("Remap Controller Buttons")
        self.map_btn.setStyleSheet("background-color: #444; padding: 10px; font-weight: bold;")
        self.map_btn.clicked.connect(self.start_mapping)
        layout.addWidget(self.map_btn)

        self.debug_chk = QCheckBox("Show Input Debug Log")
        self.debug_chk.setStyleSheet("color: #FFCC00;")
        layout.addWidget(self.debug_chk)

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

    def setup_collectors(self):
        self.kb_listener = keyboard.Listener(on_press=self.on_kb_press, on_release=self.on_kb_release)
        self.kb_listener.start()
        self.mouse_listener = mouse.Listener(on_click=self.on_mouse_click)
        self.mouse_listener.start()
        
        self.screenshot_timer = QTimer()
        self.screenshot_timer.timeout.connect(self.loop_tick)
        self.screenshot_timer.start(100) # ~10 FPS
        
        self.input_event_signal.connect(self.handle_input_event)

    # --- MAPPING & CONTROLLER ---

    def load_mapping(self):
        if self.map_path.exists():
            try:
                # Keys in JSON are always strings, convert back to int
                raw = json.load(open(self.map_path))
                self.button_map = {int(k): int(v) for k, v in raw.items()}
                print(f"Loaded {len(self.button_map)} button mappings.")
            except Exception as e:
                print(f"Map load error: {e}")

    def start_mapping(self):
        self.refresh_controllers()
        if not self.joystick:
            QMessageBox.warning(self, "Error", "No controller found to map!")
            return
            
        dlg = ControllerMapper(self.joystick, self)
        if dlg.exec_() == QDialog.Accepted:
            self.button_map = dlg.mapping
            # Save mapping
            with open(self.map_path, 'w') as f:
                json.dump(self.button_map, f)
            QMessageBox.information(self, "Success", "Mapping saved!")

    def refresh_controllers(self):
        pygame.event.pump()
        if pygame.joystick.get_count() > 0:
            if self.joystick is None:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                self.info_label.setText(f"Controller: {self.joystick.get_name()}")
        else:
            self.joystick = None
            self.info_label.setText("Controller: Disconnected")

    def loop_tick(self):
        self.refresh_controllers()
        
        v_buttons = 0
        v_lx = 0
        v_ly = 0
        
        # Flags for idle detection
        is_active_input = False
        
        if self.joystick:
            pygame.event.pump()
            
            # --- READ AXES (with Noise Gate) ---
            raw_x = self.joystick.get_axis(0)
            raw_y = self.joystick.get_axis(1)

            # Check noise gate (prevents drift from resetting idle timer)
            prev_x = self.prev_axes.get(0, 0.0)
            prev_y = self.prev_axes.get(1, 0.0)
            
            if abs(raw_x - prev_x) > IDLE_NOISE_GATE or abs(raw_y - prev_y) > IDLE_NOISE_GATE:
                is_active_input = True
                
            self.prev_axes[0] = raw_x
            self.prev_axes[1] = raw_y
            
            # Apply Deadzone for Virtual Output
            if abs(raw_x) > DEADZONE_STICK:
                v_lx = int(raw_x * 32767)
            if abs(raw_y) > DEADZONE_STICK:
                v_ly = int(raw_y * -32767) # Invert Y for Xbox standard

            # --- READ BUTTONS (with Mapping) ---
            num_buttons = self.joystick.get_numbuttons()
            pressed_physical_ids = []
            
            for i in range(num_buttons):
                if self.joystick.get_button(i):
                    pressed_physical_ids.append(i)
                    is_active_input = True
                    
                    # Apply Map
                    if i in self.button_map:
                        v_buttons |= self.button_map[i]
            
            # --- UPDATE VIRTUAL CONTROLLER ---
            self.virtual_gamepad.reset()
            self.virtual_gamepad.report.wButtons = v_buttons
            self.virtual_gamepad.report.sThumbLX = v_lx
            self.virtual_gamepad.report.sThumbLY = v_ly
            self.virtual_gamepad.update()

            # --- DEBUG LOG ---
            if self.debug_chk.isChecked() and (v_buttons > 0 or abs(v_lx) > 0):
                print(f"[DEBUG] Physical IDs: {pressed_physical_ids}")
                print(f"        Virtual Mask: {bin(v_buttons)} | LX:{v_lx} LY:{v_ly}")

        # --- IDLE LOGIC ---
        if is_active_input or len(self.active_keys) > 0:
            self.last_input_time = time.time()
            if not self.is_collecting:
                self.is_collecting = True
                self.status_label.setText("Status: Collecting")
        
        if self.is_collecting and (time.time() - self.last_input_time > self.idle_threshold):
            self.is_collecting = False
            self.status_label.setText("Status: Idle")
            self.active_keys.clear()
        
        # --- CAPTURE ---
        if self.is_collecting:
            self.record_frame(v_lx, v_ly, v_buttons)

    def record_frame(self, lx, ly, buttons):
        try:
            rect = win32gui.GetWindowRect(self.selected_window['id'])
            x, y = int(rect[0]), int(rect[1])
            w, h = int(rect[2]-rect[0]), int(rect[3]-rect[1])
            if w <= 0 or h <= 0: return

            with mss.mss() as sct:
                monitor = {"top": y, "left": x, "width": w, "height": h}
                shot = sct.grab(monitor)
                img = Image.frombytes("RGB", shot.size, shot.rgb).resize((256, 256), Image.LANCZOS)

            analog_parts = []
            if lx: analog_parts.append(f"LX:{lx/32767:.2f}")
            if ly: analog_parts.append(f"LY:{ly/32767:.2f}")

            self.frames_buffer.append(np.array(img))
            self.data_buffer.append({
                'timestamp': time.time(),
                'frame_index': len(self.frames_buffer) - 1,
                'keys': f"BtnMask_{buttons}",
                'analog': ";".join(analog_parts)
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

    def on_kb_press(self, key):
        if win32gui.GetForegroundWindow() == self.selected_window['id']:
            self.input_event_signal.emit(f"K_{key}", True)

    def on_kb_release(self, key):
        self.input_event_signal.emit(f"K_{key}", False)

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
