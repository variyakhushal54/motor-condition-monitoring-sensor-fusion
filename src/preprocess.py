"""Clean the raw recordings and add the dominant-frequency features.

For each sensor_data_*.csv file in data/, the script:
1. replaces outliers (more than 3.5 std from the median) with the median,
2. converts ax and ay to m/s^2,
3. computes the dominant frequency and its magnitude in 45-sample FFT windows.

The results are written to data/processed/. The raw files are not changed.
"""
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq

from config import (
    DATA_DIR,
    FALLBACK_SAMPLE_RATE_HZ,
    FFT_WINDOW,
    G,
    MIN_PEAK_HZ,
    MPU6050_LSB_PER_G,
    OUTLIER_SIGMA,
    PROCESSED_DIR,
)


def remove_outliers(df):
    """Replace values far from the median with the median."""
    df = df.copy()
    for col in ["sound", "ax", "ay"]:
        if col not in df.columns:
            continue
        median = df[col].median()
        std = df[col].std()
        mask = (df[col] - median).abs() > OUTLIER_SIGMA * std
        if mask.any():
            print(f"  {col}: replaced {mask.sum()} outliers with median {median:.2f}")
            df.loc[mask, col] = median
    return df


def to_metres_per_second_squared(df):
    """Convert ax and ay to m/s^2. The unit is guessed from the size of the values."""
    df = df.copy()
    mean_abs = df["ax"].abs().mean()
    if mean_abs < 20:
        print("  units: values already look like m/s^2")
        factor = 1.0
    elif mean_abs > 1000:
        print("  units: raw sensor counts (LSB)")
        factor = G / MPU6050_LSB_PER_G
    else:
        print("  units: normalised scale (0-200)")
        factor = G / 200.0
    df["ax"] = (df["ax"] * factor).abs().round(3)
    df["ay"] = (df["ay"] * factor).abs().round(3)
    return df


def estimate_sample_rate(df):
    """Estimate the sampling rate from the timestamp column, if there is one."""
    if "timestamp" in df.columns:
        median_dt = df["timestamp"].diff().dropna().median()
        if median_dt > 0:
            return 1.0 / median_dt
    return FALLBACK_SAMPLE_RATE_HZ


def add_dominant_frequency(df, fs):
    """Add frequency_hz and freq_magnitude, the peak of the FFT in each window."""
    df = df.copy()
    sound = df["sound"].to_numpy(dtype=float)
    n = len(sound)
    half = FFT_WINDOW // 2

    freq_bins = fftfreq(FFT_WINDOW, 1.0 / fs)[:half]
    valid = freq_bins > MIN_PEAK_HZ

    peak_freq = np.zeros(n)
    peak_mag = np.zeros(n)
    for i in range(n - FFT_WINDOW):
        window = sound[i : i + FFT_WINDOW]
        window = window - window.mean()          # remove the DC offset
        magnitude = np.abs(fft(window))[:half]
        if valid.any():
            k = np.argmax(np.where(valid, magnitude, -np.inf))
            peak_freq[i + half] = freq_bins[k]
            peak_mag[i + half] = magnitude[k]

    # The first and last windows have no centre value, so copy the nearest one
    peak_freq[:half] = peak_freq[half]
    peak_mag[:half] = peak_mag[half]
    last = n - half - 1
    peak_freq[last:] = peak_freq[last]
    peak_mag[last:] = peak_mag[last]

    df["frequency_hz"] = np.round(peak_freq, 2)
    df["freq_magnitude"] = np.round(peak_mag, 2)
    return df


def process_file(path):
    print(f"Processing {path.name}")
    df = pd.read_csv(path)
    df = remove_outliers(df)
    df = to_metres_per_second_squared(df)
    fs = estimate_sample_rate(df)
    print(f"  sampling rate: {fs:.2f} Hz")
    return add_dominant_frequency(df, fs)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_DIR.glob("sensor_data_*.csv"))
    if not files:
        raise SystemExit(f"No sensor_data_*.csv files found in {DATA_DIR}")
    for path in files:
        out = PROCESSED_DIR / path.name
        process_file(path).to_csv(out, index=False)
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
