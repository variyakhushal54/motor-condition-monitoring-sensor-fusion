# Data

The three raw recordings are included in this folder. The pipeline reads them from here.

Files:

| File | Scenario | Label |
|---|---|---|
| `sensor_data_healthy.csv` | motor running normally | 0 |
| `sensor_data_faulty_sound.csv` | acoustic fault (dry bearing, scraping) | 1 |
| `sensor_data_faulty_motor.csv` | mechanical fault (loose bolt, unbalanced load) | 2 |

Each file has the columns `timestamp, sound, ax, ay` (the recorded
`frequency_hz` and `freq_magnitude` columns are recomputed by `src/preprocess.py`).

- `sound`: 10-bit analog reading, 0 to 1023
- `ax`, `ay`: accelerometer values as recorded by the collector firmware
- `timestamp`: Unix time in seconds, used to estimate the sampling rate

Generated files go to `data/processed/` and are not committed, so run `src/preprocess.py` first.
