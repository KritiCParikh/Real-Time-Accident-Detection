# -*- coding: utf-8 -*-

# Initially, developed a Novice Real-Time Accident Detection System: Engineered a comprehensive model integrating hardware components (Arduino, GPS, GSM, Accelerometer) with C programming, resulting in a system capable of detecting and alerting for vehicular accidents.

Currently, I am working on a similar concept using Python with dummy data. This involves understanding additional parameters and implementing visualization to test and validate the model, aiming to expand its functionality.

# Real-time data

###### The synthetic data is generated to simulate real-time conditions, mirroring how these components would behave in a live environment. Consequently, this project focuses on real-time data processing and analysis.

# Data Collection

The dataset contains vehicle telemetry and sensor data for various analytical and testing purposes. The dataset consists of 1000 data points, each representing a minute-by-minute snapshot of vehicle conditions and operations.

Key Features:
* Timestamp: Captures the date and time of each data entry, starting from January 1, 2024.
* Speed: Measures vehicle speed in km/h, ranging from 0 to 120 km/h.
* Acceleration: Records the vehicle’s acceleration and deceleration in m/s², from -10 to 10 m/s².
* Braking: Indicates whether the brakes were applied (1) or not (0).
* GPS Coordinates: Provides latitude and longitude values for vehicle location, ranging from 20.0 to 30.0 for latitude and 80.0 to 90.0 for longitude.
* Accident Indicator: Shows if an accident occurred (1) or not (0), with a 5% occurrence rate.
* Impact Sensor: Measures impact intensity, with values between 0 and 10.
* Odometer Reading: Simulates cumulative distance traveled using a cumulative sum of random increments.
* Fuel Level: Represents fuel percentage, ranging from 10% to 100%.
* Engine Temperature: Records the engine temperature in °C, between 70 and 110 °C.
* Gyroscope Data: Provides readings along the x, y, and z axes, ranging from -5 to 5.
* Camera Images: Includes placeholder filenames for images captured by the vehicle’s camera.
* Microphone Noise Level: Measures ambient noise level in dB, between 30 and 100 dB.
* Tire Pressure: Represents tire pressure in psi, ranging from 30 to 35 psi.

This dataset is designed for developing and testing algorithms related to vehicle monitoring, accident detection, and overall automotive analytics. It provides a comprehensive view of vehicle performance and environmental conditions, enabling detailed analysis and model development.
"""

import pandas as pd
import numpy as np

# Parameters
data_size = 1000  # Number of data points

# Generate dummy data
data = {
    'timestamp': pd.date_range(start='2024-01-01', periods=data_size, freq='T'),
    'speed': np.random.uniform(0, 120, size=data_size),
    'acceleration': np.random.uniform(-10, 10, size=data_size),
    'braking': np.random.choice([0, 1], size=data_size),
    'gps_latitude': np.random.uniform(20.0, 30.0, size=data_size),
    'gps_longitude': np.random.uniform(80.0, 90.0, size=data_size),
    'accident': np.random.choice([0, 1], size=data_size, p=[0.95, 0.05]),
    'impact_sensor': np.random.uniform(0, 10, size=data_size),  # Impact sensor data
    'odometer': np.cumsum(np.random.uniform(0, 2, size=data_size)),  # Cumulative distance
    'fuel_level': np.random.uniform(10, 100, size=data_size),  # Fuel level in percentage
    'engine_temp': np.random.uniform(70, 110, size=data_size),  # Engine temperature in °C
    'gyroscope_x': np.random.uniform(-5, 5, size=data_size),  # Gyroscope x-axis data
    'gyroscope_y': np.random.uniform(-5, 5, size=data_size),  # Gyroscope y-axis data
    'gyroscope_z': np.random.uniform(-5, 5, size=data_size),  # Gyroscope z-axis data
    'camera_image': [f'image_{i}.jpg' for i in range(data_size)],  # Placeholder image filenames
    'microphone_noise_level': np.random.uniform(30, 100, size=data_size),  # Noise level in dB
    'tire_pressure': np.random.uniform(30, 35, size=data_size)  # Tire pressure in psi
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("extended_datalog.csv", index=False)

# Load the extended data
df = pd.read_csv("extended_datalog.csv")

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Summary statistics
print("Summary Statistics:")
print(df.describe(include='all'))

# Vehicle data analysis
print("\nSpeed Summary:")
print(df['speed'].describe())

print("\nAcceleration Summary:")
print(df['acceleration'].describe())

print("\nBraking Events:")
print(df['braking'].value_counts())

print("\nAccident Events:")
print(df['accident'].value_counts())

print("\nFuel Level Summary:")
print(df['fuel_level'].describe())

print("\nEngine Temperature Summary:")
print(df['engine_temp'].describe())

print("\nGyroscope Data Summary:")
print(df[['gyroscope_x', 'gyroscope_y', 'gyroscope_z']].describe())

print("\nTire Pressure Summary:")
print(df['tire_pressure'].describe())

# Camera Image Analysis (Placeholder)
print("\nCamera Image Filenames:")
print(df['camera_image'].head())

# Noise Level Analysis
print("\nMicrophone Noise Level Summary:")
print(df['microphone_noise_level'].describe())

# Timing Analysis (if timestamps are useful)
df['time_diff'] = df['timestamp'].diff().fillna(pd.Timedelta(seconds=0))
print("\nTime Differences Between Data Points:")
print(df[['timestamp', 'time_diff']].head())

# Impact Sensor Data Analysis
print("\nImpact Sensor Summary:")
print(df['impact_sensor'].describe())

# Odometer Data Analysis
print("\nOdometer Summary:")
print(df['odometer'].describe())

"""# Implement Accident Detection

This processes a CSV file to detect accidents based on predefined thresholds for impact and acceleration. It flags records with significant impact or acceleration, or where an accident is already marked, and then filters and prints these accident events with details such as timestamp, speed, and GPS coordinates.
"""

#Summary statistics
print(df[['impact_sensor', 'acceleration', 'braking']].describe())

import pandas as pd
import numpy as np

# Load the extended data
df = pd.read_csv("extended_datalog.csv")

# Define thresholds for accident detection
impact_threshold = 7  # Example threshold for impact sensor
acceleration_threshold = 8  # Example threshold for acceleration

# Detect accidents
df['detected_accident'] = ((df['impact_sensor'] > impact_threshold) |
                            (df['acceleration'].abs() > acceleration_threshold) |
                            (df['accident'] == 1)).astype(int)

# Identify accident events
accidents = df[df['detected_accident'] == 1]

# Print detected accidents
print("\nDetected Accident Events:")
print(accidents[['timestamp', 'speed', 'acceleration', 'impact_sensor', 'gps_latitude', 'gps_longitude']])

"""# Visualization and Monitoring"""

import matplotlib.pyplot as plt

# Plotting speed and acceleration data
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(df['timestamp'], df['speed'], label='Speed')
plt.title('Vehicle Speed Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Speed (km/h)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df['timestamp'], df['acceleration'], label='Acceleration', color='r')
plt.title('Vehicle Acceleration Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Acceleration (m/s^2)')
plt.legend()

plt.tight_layout()
plt.show()

"""This code builds upon previous blocks by refining the detection thresholds and enhancing the alert generation process. It consolidates the logic for detecting accidents and generating alerts into a streamlined process, ensuring accurate and clear reporting of potential accidents."""

import pandas as pd
import numpy as np

# Load the extended data
df = pd.read_csv("extended_datalog.csv")

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Define new thresholds for accident detection
impact_threshold = 7  # Adjusted threshold for impact sensor
acceleration_threshold = 5  # Adjusted threshold for acceleration
braking_threshold = 1  # Threshold for braking (binary values)

# Refined accident detection logic
df['detected_accident'] = (
    (df['impact_sensor'] > impact_threshold) |
    (df['acceleration'].abs() > acceleration_threshold) |
    (df['braking'] >= braking_threshold)
).astype(int)

# Generate alerts
def generate_alert(row):
    if row['detected_accident'] == 1:
        return f"Alert! Potential accident detected at {row['timestamp']}. Speed: {row['speed']}, Acceleration: {row['acceleration']}, Impact Sensor: {row['impact_sensor']}, GPS: ({row['gps_latitude']}, {row['gps_longitude']})"
    return None

df['alert'] = df.apply(generate_alert, axis=1)

# Filter rows with alerts
alerts = df.dropna(subset=['alert'])

# Print detected accident alerts
print("\nAccident Alerts:")
print(alerts[['timestamp', 'speed', 'acceleration', 'impact_sensor', 'gps_latitude', 'gps_longitude', 'alert']])

"""This code provides a visual summary of how often different values occur for the impact_sensor, acceleration, and braking variables in the dataset, helping to understand their distributions and patterns."""

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))

plt.subplot(3, 1, 1)
plt.hist(df['impact_sensor'], bins=30, edgecolor='k')
plt.title('Impact Sensor Distribution')
plt.xlabel('Impact Sensor Value')
plt.ylabel('Frequency')

plt.subplot(3, 1, 2)
plt.hist(df['acceleration'], bins=30, edgecolor='k')
plt.title('Acceleration Distribution')
plt.xlabel('Acceleration Value')
plt.ylabel('Frequency')

plt.subplot(3, 1, 3)
plt.hist(df['braking'], bins=2, edgecolor='k')
plt.title('Braking Distribution')
plt.xlabel('Braking Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

"""This snippet identifies and flags instances of high impact, high acceleration, and significant braking in the dataset, and then prints out these flagged instances to help analyze specific conditions where alerts were triggered."""

# Test each condition separately
df['impact_alert'] = (df['impact_sensor'] > impact_threshold).astype(int)
df['acceleration_alert'] = (df['acceleration'].abs() > acceleration_threshold).astype(int)
df['braking_alert'] = (df['braking'] >= braking_threshold).astype(int)

# Check results
print("\nImpact Alerts:")
print(df[['timestamp', 'impact_sensor', 'impact_alert']].dropna(subset=['impact_alert']))

print("\nAcceleration Alerts:")
print(df[['timestamp', 'acceleration', 'acceleration_alert']].dropna(subset=['acceleration_alert']))

print("\nBraking Alerts:")
print(df[['timestamp', 'braking', 'braking_alert']].dropna(subset=['braking_alert']))

"""This script processes vehicle data to identify and print alerts when any of the predefined thresholds are exceeded, potentially indicating an accident or near-accident scenario."""

import pandas as pd

# Load the extended data
df = pd.read_csv("extended_datalog.csv")

# Define thresholds for accident detection
impact_threshold = 7  # Example threshold for impact sensor
acceleration_threshold = 8  # Example threshold for acceleration
braking_threshold = 1  # Example threshold for braking

# Define function to check for alerts
def check_for_alert(row):
    if ((row['impact_sensor'] > impact_threshold) or
        (abs(row['acceleration']) > acceleration_threshold) or
        (row['braking'] >= braking_threshold)):
        return (f"Alert at {row['timestamp']}: High impact or acceleration detected. "
                f"Speed: {row['speed']}, Acceleration: {row['acceleration']}, "
                f"Impact Sensor: {row['impact_sensor']}, GPS: ({row['gps_latitude']}, {row['gps_longitude']})")
    return None

# Simulate real-time alert generation
for index, row in df.iterrows():
    alert = check_for_alert(row)
    if alert:
        print(alert)

"""# Future enhancements
Due to limited resources, we were unable to subscribe to services like Twilio, which facilitate real-time messaging, calls, and email interactions. Twilio is a cloud communications platform that offers APIs for integrating communication capabilities such as SMS, voice, video, and email into applications. It allows developers to programmatically send and receive messages, make phone calls, and even trigger alerts based on specific events or conditions.

In the future, incorporating Twilio into the project could significantly enhance the alert mechanism. By integrating Twilio's services, we could set up real-time notifications that are sent via SMS, email, or voice calls whenever an accident is detected or other critical conditions are met. This would enable a more responsive and immediate communication system, ensuring that alerts are delivered promptly to the appropriate contacts. Implementing and testing this component would greatly improve the overall effectiveness and reliability of the system in real-world scenarios.

Reference: Google
"""
