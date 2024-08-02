# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 23:00:47 2024

@author: 92585
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load data function
def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=' ', header=None, names=['Latitude', 'Longitude', 'Segment', 'Timestamp'])
    return df[['Latitude', 'Longitude', 'Segment']].values

# Segment paths function
def segment_paths(data):
    segments = []
    current_segment = []
    current_value = data[0, 2]
    for idx, row in enumerate(data):
        if row[2] == current_value:
            current_segment.append(row)
        else:
            segments.append(np.array(current_segment))
            current_segment = [row]
            current_value = row[2]
    segments.append(np.array(current_segment))
    return segments

# Prepare data for CNN
def prepare_data_for_cnn(data, scaler):
    features = []
    labels = []
    for segment in data:
        if len(segment.shape) == 1:  # Ensure segment is 2D
            segment = segment.reshape(1, -1)
        start = segment[0, :2]
        end = segment[-1, :2]
        passenger_status = segment[0, 2]
        segment_data = segment[:, :2]
        segment_scaled = scaler.transform(segment_data)
        
        features.append(np.concatenate([start, end, [passenger_status]]))
        labels.append(segment_scaled)
    
    features = np.array(features)
    labels = [np.array(label, dtype=np.float32) for label in labels]  # Ensure labels are float arrays
    return features, labels

# Define CNN model
def create_cnn_model(input_shape, output_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(np.prod(output_shape), activation='linear'),
        tf.keras.layers.Reshape(output_shape)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Predict function
def predict_trajectory(model, scaler, data, max_segment_length):
    predictions = []
    for segment in segment_paths(data):
        if len(segment.shape) == 1:  # Ensure segment is 2D
            segment = segment.reshape(1, -1)
        start = segment[0, :2]
        end = segment[-1, :2]
        passenger_status = segment[0, 2]
        
        features = np.concatenate([start, end, [passenger_status]]).reshape(1, -1)
        pred = model.predict(features)
        pred_rescaled = scaler.inverse_transform(pred.reshape(-1, 2)[:len(segment)])  # Ensure correct shape after prediction
        predictions.append(pred_rescaled)
    return np.concatenate(predictions, axis=0)

# Train and predict
def train_and_predict(train_data, test_data, scaler, epochs=20):
    X, y = prepare_data_for_cnn(train_data, scaler)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    input_shape = X_train.shape[1]
    max_segment_length = max(len(label) for label in y_train)
    output_shape = (max_segment_length, 2)

    y_train_padded = np.array([np.pad(label, ((0, max_segment_length - len(label)), (0, 0)), 'constant') for label in y_train])
    y_test_padded = np.array([np.pad(label, ((0, max_segment_length - len(label)), (0, 0)), 'constant') for label in y_test])

    model = create_cnn_model(input_shape, output_shape)

    model.fit(X_train, y_train_padded, epochs=epochs, batch_size=32, validation_data=(X_test, y_test_padded))

    pred = predict_trajectory(model, scaler, test_data, max_segment_length)

    return pred

# Load taxi data
taxi_files = [
    'taxi1.txt',
    'taxi2.txt',
    'taxi3.txt',
    'taxi4.txt',
    'taxi5.txt',
    'taxi6.txt'
]
taxi_data = [load_data(file) for file in taxi_files]
taxi7 = load_data('taxi7.txt')

# Normalize data
all_data = np.concatenate(taxi_data)
scaler = MinMaxScaler()
scaler.fit(all_data[:, :2])

# Train and predict
predicted_path = train_and_predict(np.concatenate(taxi_data), taxi7, scaler, epochs=20)

# 可视化总的预测路径
plt.figure(figsize=(12, 8))
plt.plot(taxi7[:, 1], taxi7[:, 0], label='True Path', color='blue')
plt.plot(predicted_path[:, 1], predicted_path[:, 0], label='Predicted Path', color='red', alpha=0.6)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('True Path vs Predicted Path')
plt.legend()
plt.show()
