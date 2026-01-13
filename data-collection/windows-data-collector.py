import sys
import time
import csv
import json
import os
import subprocess
import tempfile
import ctypes
from pathlib import Path

# --- INPUT LIBRARIES ---
import pygame
try:
    import vgamepad as vg
except ImportError:
    print("ERROR: Missing 'vgamepad'. Install it via: pip install vgamepad")
    sys.exit(1)

# --- WINDOWS & SCREEN CAPTURE ---
import win32gui
import mss
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QLabel, QLineEdit, QDialog, QListWidget, QCheckBox)
from PyQt5.QtCore import QTimer, pyqtSignal, QThread

# --- HIGH DPI FIX ---
try: ctypes.windll.shcore.SetProcessDpiAwareness(1)
except: pass

SERVER_URL = "https://neurosama.jiemonlabs.help/upload"

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
        img_tmp, csv_tmp = None, None
        try:
            # 1. Stitch Images (14x14 grid)
            grid_size = 14
            img_size = 256
            big_img = Image.new('RGB', (grid_size * img_size, grid_size * img_size), (0, 0, 0))

            for idx, frame in enumerate(self.frames):
                row, col = idx // grid_size, idx % grid_size
                big_img.paste(Image.fromarray(frame), (col * img_size, row * img_size))

            with tempfile.NamedTemporaryFile(suffix='.webp', delete=False) as f:
                big_img.save(f, format='WEBP', quality=70, method=6)
                img_tmp = f.name

            # 2. Save CSV
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp', 'frame_index', 'keys', 'analog'])
                writer.writeheader()
                writer.writerows(self.dataset)
                csv_tmp = f.name

            # 3. Upload
            cmd = ['curl', '-X', 'POST', '-H', f'X-API-KEY: {self.api_key}',
                   '-F', f'dataset=@{csv_tmp}', '-F', f'images=@{img_tmp}', SERVER_URL]
            
            res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            if res.returncode == 0: self.upload_finished.emit(True, res.stdout)
            else: self.upload_finished.emit(False, res.stderr)

        except Exception as e:
            self.upload_finished.emit(False, str(e))
        finally:
            for p in [img_tmp, csv_tmp]:
                if p and os.path.exists(p): os.unlink(p)

# ==========================================
#  GUI & COLLECTOR
# ==========================================
class APIKeyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("API Key")
        layout = QVBoxLayout()
        self.inp = QLineEdit()
        layout.addWidget(QLabel("Enter API Key:"))
        layout.addWidget(self.inp)
        btn = QPushButton("OK")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)
        self.setLayout(layout)
    def get_key(self): return self.inp.text().strip()

class WindowSelector(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Emulator Window")
        self.selected = None
        layout = QVBoxLayout()
        self.lst = QListWidget()
        layout.addWidget(self.lst)
        ref = QPushButton("Refresh")
        ref.clicked.connect(self.populate)
        layout.addWidget(ref)
        ok = QPushButton("Select")
        ok.clicked.connect(self.accept_sel)
        layout.addWidget(ok)
        self.setLayout(layout)
        self.populate()
    
    def populate(self):
        self.lst.clear()
        def cb(hwnd, l):
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
                l.append((hwnd, win32gui.GetWindowText(hwnd)))
        wins = []
        win32gui.EnumWindows(cb, wins)
        for h, t in sorted(wins, key=lambda x: x[1]): self.lst.addItem(f"{t} (ID: {h})")

    def accept_sel(self):
        i = self.lst.currentItem()
        if i:
            self.selected = {'id': int(i.text().split("ID: ")[1][:-1])}
            self.accept()

class PassthroughCollector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuro Passthrough & Collect")
        self.cfg = Path.home() / ".neuro_config.json"
        
        self.api_key = self.load_key() or self.prompt_key()
        if not self.api_key: sys.exit()

        # --- CONTROLLER SETUP ---
        pygame.init()
        pygame.joystick.init()
        self.phys_joy = None
        
        # We use DS4 (DualShock 4) because it supports generic DirectInput mapping better
        # than X360 for older emulators.
        try:
            self.virt_joy = vg.VDS4Gamepad()
            print("[System] Virtual DS4 Created.")
        except Exception as e:
            print(f"Virtual Controller Error: {e}")
            sys.exit(1)

        self.frames = []
        self.data = []
        self.batch_count = 0
        
        self.init_ui()
        self.select_win()
        if not self.target_win: sys.exit()
        
        # Start Loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.game_loop)
        self.timer.start(8) # ~120Hz polling for smooth passthrough

    def init_ui(self):
        self.setFixedSize(400, 200)
        cw = QWidget()
        self.setCentralWidget(cw)
        lay = QVBoxLayout()
        self.lbl_stat = QLabel("Status: Idle")
        lay.addWidget(self.lbl_stat)
        self.lbl_con = QLabel("Input: Searching...")
        lay.addWidget(self.lbl_con)
        self.chk_dbg = QCheckBox("Debug Inputs")
        lay.addWidget(self.chk_dbg)
        cw.setLayout(lay)

    def load_key(self):
        if self.cfg.exists(): return json.load(open(self.cfg)).get('api_key')
    
    def prompt_key(self):
        d = APIKeyDialog(self)
        if d.exec_() == QDialog.Accepted:
            k = d.get_key()
            json.dump({'api_key': k}, open(self.cfg, 'w'))
            return k

    def select_win(self):
        d = WindowSelector(self)
        if d.exec_() == QDialog.Accepted: self.target_win = d.selected
        else: self.target_win = None

    def refresh_device(self):
        """Ensures we have the PHYSICAL controller, skipping the virtual one."""
        if pygame.joystick.get_count() > 0:
            if not self.phys_joy:
                # We need to find the physical stick. 
                # Usually standard USB devices are at index 0 if plugged in first.
                for i in range(pygame.joystick.get_count()):
                    j = pygame.joystick.Joystick(i)
                    j.init()
                    # Rough filter: Don't latch onto our own Virtual controller
                    # vgamepad devices usually have specific names, but simplest is taking the first non-virtual one
                    # For now, we assume Index 0 is physical.
                    if "Xbox" not in j.get_name() and "DS4" not in j.get_name(): 
                        self.phys_joy = j
                        self.lbl_con.setText(f"Passthrough: {j.get_name()} -> Virtual DS4")
                        print(f"Latched to physical: {j.get_name()}")
                        break
                    elif i == 0: # Fallback
                         self.phys_joy = j
                         self.lbl_con.setText(f"Passthrough: {j.get_name()}")
        else:
            self.phys_joy = None
            self.lbl_con.setText("Input: Disconnected")

    def game_loop(self):
        # 1. READ PHYSICAL
        self.refresh_device()
        if not self.phys_joy: return
        
        pygame.event.pump()
        
        # --- BLIND PASSTHROUGH LOGIC ---
        # We read raw physical indices and map them 1:1 to Virtual DS4 fields.
        
        # AXES (Map -1.0...1.0 to 0...255)
        # DS4 has 6 axes usually: LX, LY, RX, RY, L2, R2
        axes = [0] * 6 
        phys_axes = self.phys_joy.get_numaxes()
        
        raw_analog_log = []
        
        for i in range(min(phys_axes, 6)):
            val = self.phys_joy.get_axis(i)
            # Normalize float (-1..1) to byte (0..255)
            byte_val = int((val + 1.0) * 127.5)
            axes[i] = byte_val
            if abs(val) > 0.1: raw_analog_log.append(f"Ax{i}:{val:.2f}")

        # Update Virtual DS4 Axes
        self.virt_joy.left_joystick_float(self.phys_joy.get_axis(0), self.phys_joy.get_axis(1))
        # Handle Extra Axes (C-buttons/Right Stick)
        if phys_axes >= 4:
            self.virt_joy.right_joystick_float(self.phys_joy.get_axis(2), self.phys_joy.get_axis(3))

        # BUTTONS (Blind Bitmask)
        # We assume Physical Btn 0 -> Virtual Btn 0 (Square), Phys Btn 1 -> Virt Btn 1 (Cross), etc.
        # This preserves the "Pressing Start is Pressing Start" logic without us knowing what "Start" is.
        phys_btns = self.phys_joy.get_numbuttons()
        virt_buttons = 0
        raw_btn_log = []
        
        # DS4 Button Map (Generic Order): 
        # Square, Cross, Circle, Triangle, L1, R1, L2, R2, Share, Opt, L3, R3, PS, Touch
        
        for i in range(min(phys_btns, 14)):
            if self.phys_joy.get_button(i):
                virt_buttons |= (1 << i) # Set the i-th bit
                raw_btn_log.append(f"B{i}")

        # Apply Buttons to Virtual Device
        # vgamepad DS4 uses specific flags, but we can inject the raw 16-bit integer if we access the report directly
        # or we just iterate and set.
        
        # Reset report
        self.virt_joy.report.wButtons = virt_buttons
        
        # Need to handle Triggers as Buttons? (Common in N64 USB)
        # Some N64 USB map Z-trigger to a button. Virtual DS4 expects Trigger as Axis (L2/R2).
        # We blindly map Button 6/7 to Triggers just in case.
        if (virt_buttons & (1 << 6)): self.virt_joy.report.bTriggerL = 255
        if (virt_buttons & (1 << 7)): self.virt_joy.report.bTriggerR = 255

        self.virt_joy.update()

        # 2. CAPTURE & UPLOAD (Only if input active)
        is_active = len(raw_btn_log) > 0 or len(raw_analog_log) > 0
        if is_active:
            self.lbl_stat.setText("Status: PASSTHROUGH ACTIVE")
            if self.chk_dbg.isChecked():
                print(f"IN: {raw_btn_log} {raw_analog_log}")
            
            # Simple Frame Limiter for Capture (10 FPS)
            if self.batch_count % 12 == 0: 
                self.capture_frame(raw_btn_log, raw_analog_log)
            self.batch_count += 1
        else:
             self.lbl_stat.setText("Status: Idle")

    def capture_frame(self, btns, axes):
        try:
            rect = win32gui.GetWindowRect(self.target_win['id'])
            x, y, x2, y2 = rect
            w, h = x2-x, y2-y
            if w<1: return
            
            with mss.mss() as sct:
                img = sct.grab({'top':y, 'left':x, 'width':w, 'height':h})
                # Convert to NP array
                frame = np.array(Image.frombytes('RGB', img.size, img.rgb).resize((256,256)))
                
            self.frames.append(frame)
            self.data.append({
                'timestamp': time.time(),
                'frame_index': len(self.frames),
                'keys': "+".join(btns),
                'analog': ";".join(axes)
            })
            
            if len(self.frames) >= 196:
                self.trigger_upload()
        except: pass

    def trigger_upload(self):
        f, d = list(self.frames), list(self.data)
        self.frames.clear()
        self.data.clear()
        UploadThread(f, d, self.api_key).start()
        print("Uploading Batch...")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = PassthroughCollector()
    w.show()
    sys.exit(app.exec_())
