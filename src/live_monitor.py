"""Read the motor rig over serial, log the readings and show live plots with an anomaly score.

Usage:
    python src/live_monitor.py              # Arduino connected to SERIAL_PORT in config.py
    python src/live_monitor.py --simulate   # generated data, for testing without hardware

The anomaly score needs models/anomaly_model.pkl and models/scaler.pkl.
Run train_anomaly_model.py first.
"""
import argparse
import csv
import math
import pickle
import queue
import threading
import time
from collections import deque

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd

from config import (
    ANOMALY_WINDOW,
    BAUD_RATE,
    FIRMWARE_SCALE,
    LIVE_LOG_FILE,
    MODEL_DIR,
    SERIAL_PORT,
)
from train_anomaly_model import ANOMALY_FEATURES

PLOT_POINTS = 200
SCORE_HISTORY = 300
WARNING_LEVEL = 0.05
CRITICAL_LEVEL = -0.05


class SerialReader:
    """Puts "sound,ax,ay" lines into a queue from a background thread."""

    def __init__(self, port, baud, simulate=False):
        self.port = port
        self.baud = baud
        self.simulate = simulate
        self.lines = queue.Queue()
        self.running = False

    def start(self):
        self.running = True
        target = self._simulate if self.simulate else self._read_serial
        threading.Thread(target=target, daemon=True).start()

    def stop(self):
        self.running = False

    def _simulate(self):
        t = 0.0
        while self.running:
            t += 0.1
            sound = int(500 + 400 * math.sin(t))
            ax = 1.5 * math.cos(t)
            ay = 1.5 * math.sin(t)
            self.lines.put(f"{sound},{ax:.2f},{ay:.2f}")
            time.sleep(0.02)

    def _read_serial(self):
        import serial  # imported here so --simulate works without pyserial

        while self.running:
            try:
                with serial.Serial(self.port, self.baud, timeout=1, dsrdtr=True) as ser:
                    time.sleep(2)  # the Arduino resets when the port opens
                    ser.reset_input_buffer()
                    print(f"Connected to {self.port}")
                    while self.running:
                        line = ser.readline().decode("utf-8", errors="ignore").strip()
                        if line:
                            self.lines.put(line)
            except Exception as exc:
                print(f"Serial error: {exc}. Retrying in 2 seconds.")
                time.sleep(2)


def parse_line(line):
    """Return (sound, ax, ay) in m/s^2, or None for header, status and bad lines."""
    if not line or line.startswith("!") or line in ("DONE", "sound,ax,ay"):
        return None
    if line.startswith("MCU"):
        print(line)
        return None
    parts = line.split(",")
    if len(parts) != 3:
        return None
    try:
        sound = int(parts[0])
        ax = float(parts[1])
        ay = float(parts[2])
    except ValueError:
        return None
    if not 0 <= sound <= 1023:  # 10-bit ADC range
        return None
    return sound, abs(ax * FIRMWARE_SCALE), abs(ay * FIRMWARE_SCALE)


class HealthMonitor:
    """Scores the latest window of readings with the saved Isolation Forest model."""

    def __init__(self):
        with open(MODEL_DIR / "anomaly_model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(MODEL_DIR / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        self.window = deque(maxlen=ANOMALY_WINDOW)
        self.scores = deque([0.0] * SCORE_HISTORY, maxlen=SCORE_HISTORY)

    def update(self, sound, ax, ay):
        self.window.append((sound, ax, ay))
        if len(self.window) < ANOMALY_WINDOW:
            return None
        df = pd.DataFrame(self.window, columns=["sound", "ax", "ay"])
        std = df.std(ddof=1)   # same as pandas rolling std used in training
        rms = (df ** 2).mean() ** 0.5
        features = pd.DataFrame(
            [list(std) + list(rms)], columns=ANOMALY_FEATURES
        )
        score = float(self.model.decision_function(self.scaler.transform(features))[0])
        self.scores.append(score)
        return score

    def status(self):
        score = self.scores[-1]
        if score >= WARNING_LEVEL:
            return "Normal"
        if score >= CRITICAL_LEVEL:
            return "Warning"
        return "Critical"


def run(simulate):
    reader = SerialReader(SERIAL_PORT, BAUD_RATE, simulate=simulate)
    monitor = None
    if (MODEL_DIR / "anomaly_model.pkl").exists():
        monitor = HealthMonitor()
    else:
        print("No model found. Showing signals without the anomaly score.")

    sound_buf = deque([0] * PLOT_POINTS, maxlen=PLOT_POINTS)
    ax_buf = deque([0.0] * PLOT_POINTS, maxlen=PLOT_POINTS)
    ay_buf = deque([0.0] * PLOT_POINTS, maxlen=PLOT_POINTS)

    LIVE_LOG_FILE.parent.mkdir(exist_ok=True)
    new_file = not LIVE_LOG_FILE.exists()
    log = open(LIVE_LOG_FILE, "a", newline="")
    writer = csv.writer(log)
    if new_file:
        writer.writerow(["timestamp", "sound", "ax", "ay"])

    fig, axes = plt.subplots(4, 1, figsize=(9, 10))
    sound_line, = axes[0].plot(range(PLOT_POINTS), sound_buf)
    ax_line, = axes[1].plot(range(PLOT_POINTS), ax_buf, color="tab:red")
    ay_line, = axes[2].plot(range(PLOT_POINTS), ay_buf, color="tab:green")
    score_line, = axes[3].plot([], [], color="tab:blue")
    axes[0].set_ylim(0, 1024)
    axes[0].set_ylabel("Sound (ADC)")
    axes[1].set_ylabel("ax (m/s^2)")
    axes[2].set_ylabel("ay (m/s^2)")
    axes[3].set_ylabel("Anomaly score")
    axes[3].axhspan(CRITICAL_LEVEL, -1, color="red", alpha=0.1)
    axes[3].axhspan(WARNING_LEVEL, CRITICAL_LEVEL, color="yellow", alpha=0.1)
    axes[3].axhspan(1, WARNING_LEVEL, color="green", alpha=0.1)
    axes[3].set_ylim(-0.3, 0.3)
    fig.tight_layout()

    def update(_frame):
        while True:
            try:
                line = reader.lines.get_nowait()
            except queue.Empty:
                break
            parsed = parse_line(line)
            if parsed is None:
                continue
            sound, ax, ay = parsed
            writer.writerow([time.time(), sound, ax, ay])
            sound_buf.append(sound)
            ax_buf.append(ax)
            ay_buf.append(ay)
            if monitor is not None:
                monitor.update(sound, ax, ay)
        log.flush()

        sound_line.set_ydata(list(sound_buf))
        ax_line.set_ydata(list(ax_buf))
        ay_line.set_ydata(list(ay_buf))
        if monitor is not None:
            scores = list(monitor.scores)
            score_line.set_data(range(len(scores)), scores)
            axes[3].set_title(f"Status: {monitor.status()}")
        return sound_line, ax_line, ay_line, score_line

    anim = animation.FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    reader.start()
    try:
        plt.show()
    finally:
        reader.stop()
        log.close()
    return anim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--simulate", action="store_true", help="use generated data instead of the serial port")
    args = parser.parse_args()
    _anim = run(args.simulate)
