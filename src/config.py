"""Shared settings for the motor condition monitoring project."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"            # raw recordings from the test rig
PROCESSED_DIR = DATA_DIR / "processed"  # output of preprocess.py and build_dataset.py
MODEL_DIR = ROOT_DIR / "models"
FIGURE_DIR = ROOT_DIR / "figures"

# Recording files and their class labels
SCENARIOS = {
    "sensor_data_healthy.csv": 0,
    "sensor_data_faulty_sound.csv": 1,
    "sensor_data_faulty_motor.csv": 2,
}
CLASS_NAMES = ["Healthy", "Faulty sound", "Faulty motor"]
FEATURE_COLUMNS = ["sound", "ax", "ay", "frequency_hz", "freq_magnitude"]

# Signal processing
OUTLIER_SIGMA = 3.5             # values further than this many std from the median are replaced
FFT_WINDOW = 45                 # samples per FFT window
FALLBACK_SAMPLE_RATE_HZ = 45.0  # used when there is no timestamp column
MIN_PEAK_HZ = 1.0               # spectral peaks below this frequency are ignored

# Unit conversion
G = 9.81                        # m/s^2
MPU6050_LSB_PER_G = 16384.0     # sensitivity at the +/-2 g range

# Anomaly detection
ANOMALY_WINDOW = 50
ANOMALY_CONTAMINATION = 0.01

# Live monitor
SERIAL_PORT = "COM13"           # change this to the port of your Arduino
BAUD_RATE = 115200
LIVE_LOG_FILE = DATA_DIR / "live_log.csv"
# The firmware divides the accelerometer values by 200 before sending them.
# The live monitor multiplies by this factor to get back to m/s^2.
# This does not fully match the recorded data yet, see the README.
FIRMWARE_SCALE = G / 200.0
