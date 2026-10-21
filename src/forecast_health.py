"""Forecast vibration with ARIMA and estimate when the health index crosses 20 %.

The degradation is simulated. The healthy vibration signal gets a linear drift
that starts at sample 3500. No real wear data is available yet, so the result
shows how the method works. It does not say how long a real motor will last.

One sample is treated as one hour, so 24 samples make one day.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.arima.model import ARIMA

from config import FIGURE_DIR, PROCESSED_DIR

TOTAL_SAMPLES = 5000
DRIFT_START = 3500
DRIFT_PER_SAMPLE = 0.0010   # m/s^2 added per sample after DRIFT_START
SMOOTH_WINDOW = 24          # moving average, one day of samples
SAMPLES_PER_DAY = 24
FORECAST_DAYS = 25
HISTORY_LENGTH = 1000       # samples used to fit ARIMA
VIB_NOMINAL = 0.6           # m/s^2, mapped to 100 % health
VIB_FAILURE = 5.5           # m/s^2, mapped to 0 % health
FAILURE_HEALTH = 20.0       # percent


def build_signal():
    """Healthy signal repeated to 5000 samples, plus a linear drift, smoothed."""
    healthy = pd.read_csv(PROCESSED_DIR / "sensor_data_healthy.csv")
    base = healthy["ax"].to_numpy()
    repeats = TOTAL_SAMPLES // len(base) + 1
    signal = np.tile(base, repeats)[:TOTAL_SAMPLES]

    drift = np.zeros(TOTAL_SAMPLES)
    drift[DRIFT_START:] = np.arange(TOTAL_SAMPLES - DRIFT_START) * DRIFT_PER_SAMPLE

    smooth = pd.Series(signal + drift).rolling(SMOOTH_WINDOW).mean().dropna()
    return smooth.to_numpy()


def health_percent(vibration):
    """Linear map from vibration level to health: 100 % at nominal, 0 % at failure."""
    clipped = np.clip(vibration, VIB_NOMINAL, VIB_FAILURE)
    normalised = (clipped - VIB_NOMINAL) / (VIB_FAILURE - VIB_NOMINAL)
    return (1.0 - normalised) * 100.0


def main():
    FIGURE_DIR.mkdir(exist_ok=True)
    values = build_signal()

    horizon = FORECAST_DAYS * SAMPLES_PER_DAY
    history = values[:-horizon]
    held_out = values[-horizon:]

    model = ARIMA(history[-HISTORY_LENGTH:], order=(5, 1, 0), trend="t").fit()
    forecast = model.forecast(steps=horizon)

    mae = np.mean(np.abs(forecast - held_out))
    print(f"Mean absolute error on the held-out {FORECAST_DAYS} days: {mae:.3f} m/s^2")

    health = gaussian_filter1d(health_percent(forecast), sigma=5)
    crossing = np.where(health < FAILURE_HEALTH)[0]
    if len(crossing):
        days = crossing[0] / SAMPLES_PER_DAY
        print(f"Forecast health drops below {FAILURE_HEALTH:.0f} % after about {days:.1f} days")
    else:
        print(f"Forecast stays above {FAILURE_HEALTH:.0f} % health for {FORECAST_DAYS} days")

    t_history = np.arange(len(history)) / SAMPLES_PER_DAY
    t_forecast = (len(history) + np.arange(horizon)) / SAMPLES_PER_DAY

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t_history, history, color="green", alpha=0.6, label="Simulated vibration")
    ax.plot(t_forecast, held_out, color="grey", alpha=0.6, label="Held-out simulated vibration")
    ax.plot(t_forecast, forecast, "--", color="red", label="ARIMA forecast")
    ax.set_xlabel("Time (days, one sample = one hour)")
    ax.set_ylabel("Vibration (m/s^2)")
    ax.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(t_forecast, health, color="blue", linewidth=2, label="Health index (%)")
    ax2.axhline(FAILURE_HEALTH, color="purple", linestyle=":", label="Failure threshold")
    ax2.set_ylabel("Health index (%)")
    ax2.set_ylim(0, 110)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")
    ax.set_title(f"Simulated degradation: ARIMA(5,1,0) forecast, {FORECAST_DAYS}-day horizon")
    fig.tight_layout()
    out = FIGURE_DIR / "health_forecast.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
