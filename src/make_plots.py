"""Exploratory plots for the three recording scenarios."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import CLASS_NAMES, FEATURE_COLUMNS, FIGURE_DIR, PROCESSED_DIR, SCENARIOS

PALETTE = {"Healthy": "green", "Faulty sound": "red", "Faulty motor": "orange"}


def load_scenarios():
    frames = []
    for filename, label in SCENARIOS.items():
        df = pd.read_csv(PROCESSED_DIR / filename)
        df["Scenario"] = CLASS_NAMES[label]
        df["Sample"] = range(len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_signals_over_time(data):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, col, label in zip(
        axes,
        ["frequency_hz", "ax", "ay"],
        ["Dominant frequency (Hz)", "ax (m/s^2)", "ay (m/s^2)"],
    ):
        sns.lineplot(data=data, x="Sample", y=col, hue="Scenario", palette=PALETTE,
                     alpha=0.7, errorbar=None, ax=ax)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("Sample index")
    axes[0].set_title("Signals per scenario")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "signals_over_time.png", dpi=150)
    plt.close(fig)


def plot_correlation(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = data[FEATURE_COLUMNS].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Feature correlation")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "feature_correlation.png", dpi=150)
    plt.close(fig)


def main():
    FIGURE_DIR.mkdir(exist_ok=True)
    data = load_scenarios()
    plot_signals_over_time(data)
    plot_correlation(data)
    print(f"Saved plots to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
