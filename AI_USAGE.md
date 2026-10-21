# AI usage disclosure

This project was written by Khushalkumar Variya as a course project at OTH
Amberg-Weiden. An AI assistant (Claude) was used later for the tasks listed
below. The author reviewed and decided on each of them. Work the assistant did
not do is listed too.

## Used for

- Reorganising the Python scripts into `src/` and adding a shared `config.py`.
- Changing the preprocessing scripts to write to `data/processed/`.
- Adding a mean absolute error check on the 25 held-out forecast days.
- Writing and tidying documentation: `README.md`, `data/README.md`, the text
  version of the report in `docs/REPORT.md`, `requirements.txt` and `.gitignore`.
- Adding the results table and the list of differences between the report and
  the code in `README.md`.

## Not done by the assistant

- Data collection, the hardware setup and the fault scenarios.
- Running the full training pipeline or the ARIMA forecast. The results quoted
  in the README come from the final report and were not re-run.
- Flashing or testing the Arduino firmware.
- Any change to the report PDF.

## Checks

- The Python files were checked for syntax errors.
- Some steps were run on test copies to confirm the code executes. Those outputs
  are not reported as results.
