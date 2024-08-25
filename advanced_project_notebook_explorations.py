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

"""# Metrics"""

import pandas as pd

# Load the data
df = pd.read_csv("extended_datalog.csv")

# Summary statistics
print(df.describe(include='all'))

mean_speed = df['speed'].mean()
print(f"Mean Speed: {mean_speed}")

percentile_95_speed = df['speed'].quantile(0.95)
print(f"95th Percentile of Speed: {percentile_95_speed}")

# Count occurrences of braking events
braking_count = df['braking'].sum()
print(f"Total Braking Events: {braking_count}")

# Count occurrences of accidents
accidents_count = df['accident'].sum()
print(f"Total Accidents: {accidents_count}")

# Calculate detected accidents based on thresholds
impact_threshold = 7
acceleration_threshold = 8

df['detected_accident'] = ((df['impact_sensor'] > impact_threshold) |
                            (df['acceleration'].abs() > acceleration_threshold) |
                            (df['accident'] == 1)).astype(int)

# Count detected accidents
detected_accidents_count = df['detected_accident'].sum()
print(f"Total Detected Accidents: {detected_accidents_count}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(df['timestamp'], df['speed'])
plt.title('Vehicle Speed Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Speed')
plt.show()

# Define thresholds for alerts
impact_threshold = 7
acceleration_threshold = 8
braking_threshold = 1

# Generate alerts
def generate_alert(row):
    if (row['impact_sensor'] > impact_threshold or
        abs(row['acceleration']) > acceleration_threshold or
        row['braking'] >= braking_threshold):
        return f"Alert! High impact or acceleration detected at {row['timestamp']}"
    return None

df['alert'] = df.apply(generate_alert, axis=1)

# Count alerts
alerts_count = df['alert'].notna().sum()
print(f"Total Alerts: {alerts_count}")

# Load the data
df = pd.read_csv("extended_datalog.csv")

# Print column names
print(df.columns)

#Column creation
import pandas as pd
import numpy as np

# Load the extended data
df = pd.read_csv("extended_datalog.csv")

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Define thresholds for accident detection
impact_threshold = 7  # Example threshold for impact sensor
acceleration_threshold = 8  # Example threshold for acceleration
braking_threshold = 1  # Threshold for braking (binary values)

# Detect accidents
df['detected_accident'] = (
    (df['impact_sensor'] > impact_threshold) |
    (df['acceleration'].abs() > acceleration_threshold) |
    (df['braking'] >= braking_threshold)
).astype(int)

# Save updated DataFrame to CSV (optional)
df.to_csv("extended_datalog.csv", index=False)

# Print column names to confirm
print(df.columns)

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Reload the updated data
df = pd.read_csv("extended_datalog.csv")

# True Labels (Ground Truth) and Predicted Labels
true_labels = df['accident']
predicted_labels = df['detected_accident']

# Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)
tn, fp, fn, tp = cm.ravel()  # Extract values from confusion matrix

# Calculate metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
false_positive_rate = fp / (fp + tn)
false_negative_rate = fn / (fn + tp)

# Print metrics
print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"False Positive Rate: {false_positive_rate:.2f}")
print(f"False Negative Rate: {false_negative_rate:.2f}")

"""Model is predicting a lot of false positives, leading to low precision and high false positive rate. This suggests a class imbalance, with a small number of positive cases compared to negative ones.

# Recommendations for Improvement

1.Data Preprocessing:

* Feature Engineering

* Handling Imbalanced Data


2. Model Tuning:

* Hyperparameter Optimization

* Different Algorithms


3. Model Evaluation:

* Cross-Validation

* Adjust Thresholds


4. Feature Selection:

* Remove Irrelevant Features

* Feature Importance

By addressing these areas, we would be able to improve the model’s performance and achieve better balance between precision and recall.

# Trial: Feature Engineering and Data Preparation
"""

print(df.columns)

"""1. Ensure Correct Encoding of Categorical Features
Since the dataset contains categorical features like camera_image and microphone_noise_level, we need to encode them properly before applying resampling.

2. Update the Code for Preprocessing and Resampling
Below is an updated version of the code that includes preprocessing for categorical features and ensures proper handling of these features:
"""

df['speed_change'] = df['speed'].diff().fillna(0)
df['acceleration_change'] = df['acceleration'].diff().fillna(0)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['speed', 'acceleration', 'impact_sensor', 'fuel_level', 'engine_temp', 'gyroscope_x', 'gyroscope_y', 'gyroscope_z']] = scaler.fit_transform(
    df[['speed', 'acceleration', 'impact_sensor', 'fuel_level', 'engine_temp', 'gyroscope_x', 'gyroscope_y', 'gyroscope_z']]
)

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['hour'] = df['timestamp'].dt.hour
df.drop(columns=['timestamp'], inplace=True)

print(df.columns)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the data
df = pd.read_csv("extended_datalog.csv")

# Drop non-numeric columns (e.g., 'camera_image')
df = df.drop(columns=['camera_image'])

# Convert datetime columns to numeric (if necessary)
# Assuming 'timestamp' is in datetime format, but you may need to convert it if it's not
# df['timestamp'] = pd.to_datetime(df['timestamp'])
# Extract features from datetime if needed (e.g., year, month, day, hour)
# df['year'] = df['timestamp'].dt.year
# df['month'] = df['timestamp'].dt.month
# df['day'] = df['timestamp'].dt.day
# df['hour'] = df['timestamp'].dt.hour

# Drop columns that may still be non-numeric or irrelevant
df = df.select_dtypes(include=[float, int])

# Features and target variable
X = df.drop(columns=['accident'])
y = df['accident']

# Handle missing values (if any)
X = X.fillna(0)  # or use another strategy for missing values

# Initialize and fit the RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_

# Sort features by importance
feature_names = X.columns
sorted_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

print("Feature Importances:")
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")

"""# How Feature Importances Improve Model Performance
Feature Selection:

Prioritize Key Features

Feature Engineering

Enhance and Transform Features

Data Preparation

Optimize Scaling and Transformation

Handling Imbalanced Data

Balanced Training

Model Tuning

Hyperparameter Optimization

Exploratory Data Analysis (EDA)

Understand Feature Relationships

Model Interpretation

Explain Predictions

By leveraging feature importances to guide feature selection, engineering, and model tuning, we can significantly enhance model accuracy and effectiveness.
"""

# Make predictions
y_pred = best_model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Accident', 'Accident'],
            yticklabels=['No Accident', 'Accident'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

"""# Analysis:

* High Performance on Class 0: The model is highly accurate in predicting the absence of an accident (Class 0) with a precision of 0.95 and recall of 0.92.
* Low Performance on Class 1: The model struggles to predict the presence of an accident (Class 1), with a precision of 0.08 and recall of 0.13. This indicates a high number of false negatives for Class 1.

# Suggestions for Improvement:

1. Address Class Imbalance:

* Resampling: Use techniques like SMOTE (Synthetic Minority Over-sampling Technique) or ADASYN to balance the class distribution.
* Class Weights

2. Model Refinement:

* Try Different Algorithms: Test other classifiers like Gradient Boosting, XGBoost, or Logistic Regression, which might handle imbalanced data better.

* Hyperparameter Tuning


3. Feature Engineering:

* Create More Features: Add more relevant features or perform feature interactions that might better capture patterns related to accidents.

* Feature Selection

4. Cross-Validation:

* Use Stratified K-Fold
"""

from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imPipeline

# Create SMOTE object
smote = SMOTE(random_state=42)

# Define updated pipeline with SMOTE
pipeline = imPipeline([
    ('scaler', scaler),
    ('smote', smote),
    ('feature_selection', selector),
    ('classifier', RandomForestClassifier())
])

# Perform GridSearchCV with updated pipeline
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Accident', 'Accident'],
            yticklabels=['No Accident', 'Accident'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

"""# Suggestions for Improvement

* Fine-Tune SMOTE:

* Try Different Resampling Techniques:

* Optimize Model Complexity:

* Feature Engineering and Selection

* Review Features

* Dimensionality Reduction

* Cross-Validation and Evaluation Metrics:

* Stratified K-Fold Cross-Validation

* Other Evaluation Metrics

# Future enhancements
Due to limited resources, we were unable to subscribe to services like Twilio, which facilitate real-time messaging, calls, and email interactions. Twilio is a cloud communications platform that offers APIs for integrating communication capabilities such as SMS, voice, video, and email into applications. It allows developers to programmatically send and receive messages, make phone calls, and even trigger alerts based on specific events or conditions.

In the future, incorporating Twilio into the project could significantly enhance the alert mechanism. By integrating Twilio's services, we could set up real-time notifications that are sent via SMS, email, or voice calls whenever an accident is detected or other critical conditions are met. This would enable a more responsive and immediate communication system, ensuring that alerts are delivered promptly to the appropriate contacts. Implementing and testing this component would greatly improve the overall effectiveness and reliability of the system in real-world scenarios.

# Data Preprocessing

Resampling Techniques: Using techniques to balance the dataset.

Oversampling: Increase the number of accident cases (positive class) using methods like SMOTE (Synthetic Minority Over-sampling Technique).

Undersampling: Reduce the number of non-accident cases (negative class) to balance the dataset.

Reference: Google
"""
