# Hardware Sensor Implementation and Predictive Motor Condition Analysis

Khushalkumar Variya
OTH Amberg-Weiden, Study programme "Artificial Intelligence for Industrial Applications"

## Abstract

Unplanned motor breakdowns are expensive. This project builds a motor health
monitoring system that combines two sensors: an MPU6050 accelerometer for
vibration and a microphone for sound. The system computes features with the Fast
Fourier Transform (FFT) and rolling windows, such as RMS energy and dominant
frequency. An Isolation Forest detects anomalies using only healthy-motor data.
A supervised classifier (Random Forest or Gradient Boosting) separates healthy,
faulty-sound and faulty-motor states. An ARIMA model forecasts a simulated
degradation trend. A Python dashboard shows the sensor streams, a spectrogram and
a health score in real time.

## 1. Introduction

Single-sensor systems miss some faults. Vibration sensing picks up imbalance and
loose parts, while sound sensing picks up dry bearings and scraping. This project
combines both. Isolation Forest was chosen for real-time anomaly detection because
it is light enough for edge hardware and does not need many labelled failures
[1], [4]. ARIMA is used for trend forecasting [3]. Sensor fusion for motor
fault detection is discussed in [2] and [5].

## 2. Hardware

### 2.1 Sensors

- **Accelerometer (MPU6050):** measures acceleration on the X and Y axes.
  Configured for the ±2 g range, which gives 16384 counts per g.
- **Microphone:** an analog sound sensor read by a 10-bit ADC on pin A0.
  Values range from 0 to 1023.

The firmware divides the accelerometer counts by 200 before sending them over
serial. See Section 6.1 for the effect on units.

### 2.2 Data acquisition

An Arduino runs the motors through an L298N driver and streams one line per
reading over USB at 115200 baud. Each line has the form `sound,ax,ay`. Sampling is
about 50 Hz. The PC side reads the serial port in a background thread, so the
plotting window stays responsive.

### 2.3 Connection handling

The live monitor retries the serial connection if the cable is unplugged, and
reconnects when it is plugged back in.

## 3. Test scenarios

Three recordings were made on the same rig:

1. **Healthy:** the motor runs normally. Vibration is low and sound is steady.
   File: `sensor_data_healthy.csv`.
2. **Faulty motor:** mechanical faults are set up by hand, such as a loose bolt
   or an unbalanced rotor. Accelerometer values are much larger than in the
   healthy recording. File: `sensor_data_faulty_motor.csv`.
3. **Faulty sound:** a fault that can be heard but is not yet felt, such as a dry
   bearing. The sound signal has high spikes while vibration stays low.
   File: `sensor_data_faulty_sound.csv`.

## 4. Data collection and storage

The firmware sends 10,000 readings per run. The microphone gives a continuous
10-bit stream. The MPU6050 gives 3-axis motion data, of which only X and Y are
used. The live monitor writes every reading to a CSV file with a timestamp.

Noise is handled by the sensor hardware. An earlier version of the code also added
random noise on purpose for testing. That option was removed from the repository.

## 5. Software architecture

### 5.1 Pipeline

The live monitor is built as a chain of nodes. Each node has a `process` method and
passes its output to the nodes connected after it:

- **Serial source:** reads lines from the Arduino.
- **CSV writer:** logs each reading to disk.
- **Live plotter:** keeps a rolling buffer for the plots.
- **Health monitor:** scores each window with the Isolation Forest.

### 5.2 Concurrency

A background thread reads the serial port and puts lines on a thread-safe queue.
The main thread takes lines off the queue and updates the plots, so blocking I/O
does not freeze the window.

### 5.3 Health status

The health monitor scores the latest 50 readings. The score is compared with two
thresholds to give a status:

- **Normal:** score at or above 0.05
- **Warning:** score between -0.05 and 0.05
- **Critical:** score below -0.05

### 5.4 Spectrogram

The sound signal is shown as a rolling histogram over time. The x-axis is time,
the y-axis is the amplitude band, and the colour is the count in each band.

## 6. Feature engineering

### 6.1 Unit conversion

**Sound.** The 10-bit ADC gives values from 0 to 1023. Silence sits at about 512.
The FFT step removes this DC offset.

**Accelerometer.** The MPU6050 is set to ±2 g, so 1 g corresponds to 16384 counts.
The firmware divides by 200, so 1 g becomes about 82 in the serial data. The Python
code then multiplies by 9.81/200. As a result, 1 g ends up at about 4.0 m/s²
instead of 9.81 m/s².

### 6.2 Standardisation

Features are standardised before they go into the models so that large numbers do
not dominate small ones. Models that use distances (such as the Isolation Forest)
require this. A StandardScaler is fitted on the training data and saved with the
model.

### 6.3 Outliers

Values more than 3.5 standard deviations from the median are replaced by the
median. This removes single-sample spikes from sensor transients.

### 6.4 FFT dominant frequency

For each sound sample, the script takes a window of 45 samples (about 0.9 s at
50 Hz), removes the mean, and computes the FFT. The frequency with the largest
magnitude above 1 Hz is the dominant frequency. The magnitude at that frequency is
also stored. The first and last windows have no centre sample, so they copy their
nearest neighbour.

The sampling rate is estimated from the timestamp column. If there is no timestamp,
45 Hz is used.

### 6.5 Final feature set

The classifiers use five features: `sound`, `ax`, `ay`, `frequency_hz`,
`freq_magnitude`. The timestamp is kept in the CSV but not used as a feature.

*Correction:* the original report listed seven features including the timestamp
and the label. The label is the target, not a feature.

## 7. Dataset preparation

1. **Clean:** outlier removal and unit conversion (Section 6).
2. **Label:** healthy = 0, faulty sound = 1, faulty motor = 2.
3. **Combine:** the three recordings are concatenated into one dataset.
4. **Split:** 80 % for training and 20 % as a hold-out set, stratified by class.

**Class balance.** Healthy data is usually more common in practice. Isolation
Forest handles this by learning what normal looks like, so it does not need
balanced classes.

## 8. Machine learning

### 8.1 Models

- **Isolation Forest (unsupervised):** 100 trees, contamination 0.01, with a
  StandardScaler. Trained on rolling-window features (standard deviation and RMS
  over 50 samples) from the healthy recording. Used for live anomaly scoring.
- **Random Forest (200 trees) and Gradient Boosting (200 estimators, learning
  rate 0.1, max depth 5):** supervised classifiers for the three states. The
  better one on validation data is chosen.

### 8.2 Model selection

The training data is split again into 80 % fit and 20 % validation. Both
classifiers are fitted and the one with the higher validation accuracy is kept.
Ties go to Random Forest.

### 8.3 Saving the models

The Isolation Forest and its scaler are saved as pickle files so the live monitor
can load them without retraining.

## 9. Evaluation

### 9.1 Visual checks

- Signals over time for each scenario.
- Feature correlation heatmap.

### 9.2 Hold-out results

On the 20 % hold-out set, Gradient Boosting reached **99.27 % accuracy** and a
weighted F1 of about 0.99, as reported in the original report.

*Caveat:* the hold-out set is a random split by row. Neighbouring rows come from
overlapping windows of the same recording, so the score is optimistic. A split by
recording block would be a fairer test and is the planned next step.

## 10. Time series forecasting

### 10.1 Simulated degradation

The healthy vibration signal is repeated to 5000 samples. From sample 3500 onward,
a linear drift of 0.001 per sample is added. The signal is smoothed with a 24-sample
moving average. One sample is treated as one hour.

The degradation is **simulated**. No real wear data is available. The forecast
therefore shows how the method works, not how long a real motor will last.

### 10.2 ARIMA model

An ARIMA(5,1,0) model with a linear trend is fitted to the last 1000 samples before
the final 25 days. It forecasts the 25 days.

- **d = 1:** first-order differencing removes the linear trend, which makes the
  series stationary.
- **p = 5:** the 24-sample smoothing creates strong autocorrelation, so five lags
  are used.
- **q = 0:** the trend is handled by the AR terms, so moving-average terms are not
  added.

The forecast is compared with the held-out 25 days using mean absolute error. The
original report did not include this comparison.

### 10.3 Health index

Vibration is mapped linearly to a health percentage: 0.6 m/s² is 100 % and 5.5 m/s²
is 0 %. A Gaussian filter smooths the result. The estimated time at which the
health index drops below 20 % is printed as the time to failure.

## 11. Validation and deployment

- The final classifier is evaluated on the hold-out set (Section 9.2).
- The live monitor can run without hardware using `--simulate`.
- The live monitor runs on a laptop. The original report mentioned a Raspberry Pi
  Zero, but this was not tested.

## 12. Conclusion

The project shows that a low-cost sensor setup, combining an accelerometer and a
microphone, can separate healthy and faulty motor states on the recordings made
here. Isolation Forest gives a real-time anomaly score and an ARIMA model
illustrates a forecast.

The main limits are: only three recordings from one rig, a random row-level split,
unit conversion that is about 2.4 times off in absolute terms, and a forecast based
on simulated drift. These should be addressed before the results are used beyond
this prototype.

## References

[1] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in *2008 Eighth IEEE
International Conference on Data Mining*, 2008, pp. 413–422.

[2] X. Li, W. Zhang, and Q. Ding, "A robust approach of multi-sensor fusion for fault
diagnosis using convolution neural network," in *Proceedings of the IEEE International
Conference on Industrial Technology (ICIT)*, 2019.

[3] S. Sakhare and S. Wagh, "ARIMA model based real time trend analysis for predictive
maintenance," in *2019 International Conference on Computing, Power and Communication
Technologies (GUCON)*, IEEE, 2019.

[4] M. A. Alsoufi and S. Alraddadi, "Isolation forest based anomaly detection: A
systematic literature review," *IEEE Access*, vol. 9, 2021.

[5] P. A. Delgado-Arredondo, D. Morinigo-Sotelo, and R. A. Osornio-Rios, "Acoustic
analysis based condition monitoring of induction motors: A review," *Measurement*,
vol. 191, 2022.

[6] D. Fawwaz and A. B. Mutiara, "Time-series classification methods: Review and
applications to power systems data," *IEEE Access*, vol. 10, 2022.

[7] A. Bouchareb and S. Hrairi, "AI-based predictive maintenance for machine vibration
and condition monitoring," *Journal of Quality in Maintenance Engineering*, 2023.

[8] A. Glowacz and R. Tadeusiewicz, "The evolution of machine learning in vibration and
acoustics: A decade of innovation (2015–2024)," *Applied Sciences*, vol. 15, no. 12, 2024.

[9] J. Silva and A. Santos, "Enhancing three-phase induction motor reliability with
health index and artificial intelligence-driven predictive maintenance," *Scientific
Reports*, vol. 14, 2024.
