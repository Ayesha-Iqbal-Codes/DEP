#!/usr/bin/env python
# coding: utf-8

# # Data Collection and Preprocessing

# In[2]:


import pandas as pd

file_path = 'CloudWatch_Traffic_Web_Attack.csv'
df = pd.read_csv(file_path)

df['creation_time'] = pd.to_datetime(df['creation_time'])
df['end_time'] = pd.to_datetime(df['end_time'])
df['time'] = pd.to_datetime(df['time'])

missing_values = df.isnull().sum()
print(df.info())
print(missing_values)


# # Feature Extraction

# In[3]:


df['packet_size'] = df['bytes_in'] + df['bytes_out']

df['duration'] = (df['end_time'] - df['creation_time']).dt.total_seconds()

df['time_interval'] = df['time'].diff().dt.total_seconds().fillna(0)

print(df.head())


# # Anomaly Detection with Isolation Forest

# In[4]:


from sklearn.ensemble import IsolationForest

features = ['packet_size', 'duration', 'time_interval']

isolation_forest = IsolationForest(contamination=0.1, random_state=42)
df['anomaly_if'] = isolation_forest.fit_predict(df[features])

df['anomaly_if'] = df['anomaly_if'].apply(lambda x: 'anomaly' if x == -1 else 'normal')

anomaly_counts_if = df['anomaly_if'].value_counts()
print(df.head())
print(anomaly_counts_if)


# # Anomaly Detection with Autoencoders

# In[5]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

X = df[features].values
X = (X - X.mean(axis=0)) / X.std(axis=0)  

input_dim = X.shape[1]
encoding_dim = 2

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="linear")(encoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.compile(optimizer="adam", loss="mean_squared_error")
autoencoder.fit(X, X, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

reconstructions = autoencoder.predict(X)
reconstruction_errors = np.mean((X - reconstructions) ** 2, axis=1)

threshold = np.percentile(reconstruction_errors, 90)
df['anomaly_ae'] = reconstruction_errors > threshold

df['anomaly_ae'] = df['anomaly_ae'].apply(lambda x: 'anomaly' if x else 'normal')

anomaly_counts_ae = df['anomaly_ae'].value_counts()
print(df.head())
print(anomaly_counts_ae)


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='packet_size', y='duration', hue='anomaly_if', palette=['blue', 'red'])
plt.title('Packet Size vs. Duration (Isolation Forest)')
plt.xlabel('Packet Size')
plt.ylabel('Duration (seconds)')
plt.legend(title='Anomaly')
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(df['time'], df['packet_size'], label='Packet Size')
anomalies = df[df['anomaly_if'] == 'anomaly']
plt.scatter(anomalies['time'], anomalies['packet_size'], color='red', label='Anomaly')
plt.title('Time Series of Packet Sizes with Anomalies')
plt.xlabel('Time')
plt.ylabel('Packet Size')
plt.legend()
plt.show()

features = ['packet_size', 'duration', 'time_interval']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, feature in enumerate(features):
    sns.histplot(data=df, x=feature, hue='anomaly_if', multiple="stack", ax=axes[i])
    axes[i].set_title(f'Histogram of {feature}')
plt.show()


# In[ ]:




