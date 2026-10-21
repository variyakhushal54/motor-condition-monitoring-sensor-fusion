"""Combine the processed recordings, add labels and split into train and hold-out sets.

Outputs in data/processed/:
- combined_data_with_status.csv  all rows with a Status label
- training_data.csv              80 % (stratified by class)
- unseen_data.csv                20 % hold-out set

Note: the split is done on single rows. Neighbouring rows come from overlapping
windows, so the hold-out score is optimistic. See the README.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from config import PROCESSED_DIR, SCENARIOS


def main():
    frames = []
    for filename, label in SCENARIOS.items():
        df = pd.read_csv(PROCESSED_DIR / filename)
        df["Status"] = label
        print(f"{filename}: {len(df)} rows, label {label}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(PROCESSED_DIR / "combined_data_with_status.csv", index=False)

    train, holdout = train_test_split(
        combined, test_size=0.2, random_state=42, stratify=combined["Status"]
    )
    train.to_csv(PROCESSED_DIR / "training_data.csv", index=False)
    holdout.to_csv(PROCESSED_DIR / "unseen_data.csv", index=False)
    print(f"Training rows: {len(train)}, hold-out rows: {len(holdout)}")


if __name__ == "__main__":
    main()
