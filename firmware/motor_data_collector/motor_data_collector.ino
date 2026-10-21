/*
 * Motor data collector (Arduino firmware)
 *
 * Runs two DC motors through an L298N driver at a fixed speed. Reads an MPU6050
 * accelerometer (X and Y axes) and an analog microphone, and sends one line per
 * reading over serial in the form:  sound,ax,ay
 *
 * The header line "sound,ax,ay" is sent before the readings and "DONE" after the
 * last one. Sampling is about 50 Hz (20 ms delay per reading).
 */

#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

// L298N motor driver pins
const int ENA = 5;    // PWM speed, left motor
const int IN1 = 7;
const int IN2 = 8;
const int ENB = 6;    // PWM speed, right motor
const int IN3 = 11;
const int IN4 = 9;

const int MIC_PIN = A0;

// PWM value from 0 to 255. Lower values keep the USB supply stable under load.
const int MOTOR_SPEED = 150;

const int NUM_READINGS = 10000;
const int SAMPLE_DELAY_MS = 20;

void stopMotors() {
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
}

void setup() {
  delay(2000);                 // give the PC time to detect the USB device
  Serial.begin(115200);
  while (!Serial) delay(10);   // needed on Leonardo and Micro boards

  Serial.println("MCU: starting");

  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  // Motors off until the sensor is confirmed working
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  stopMotors();

  Wire.begin();
  mpu.initialize();

  if (!mpu.testConnection()) {
    Serial.println("!ERROR: MPU6050 not found. Check SDA/SCL wiring and power.");
    while (1);                 // stop here so no empty data is recorded
  }
  Serial.println("MCU: MPU6050 ready");
  delay(1000);                 // let the sensor settle

  // Forward direction for both motors
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  analogWrite(ENA, MOTOR_SPEED);
  analogWrite(ENB, MOTOR_SPEED);
  Serial.println("MCU: motors running");

  delay(500);
  Serial.println("sound,ax,ay");
}

void loop() {
  for (int i = 0; i < NUM_READINGS; i++) {
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    int sound = analogRead(MIC_PIN);

    // Divide by 200 so the numbers are small enough to read in the log
    float axScaled = ax / 200.0;
    float ayScaled = ay / 200.0;

    Serial.print(sound);
    Serial.print(',');
    Serial.print(axScaled);
    Serial.print(',');
    Serial.println(ayScaled);

    delay(SAMPLE_DELAY_MS);
  }

  Serial.println("DONE");
  stopMotors();
  while (1);                   // wait for a reset
}
