import yfinance as yf
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

TICKER = 'AAPL'  # or TSLA, MSFT etc.

df = yf.download(TICKER, start='2015-01-01')
df = df[['Close']]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

seq_len = 60
X, y = [], []
for i in range(seq_len, len(scaled_data)):
    X.append(scaled_data[i-seq_len:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=5, batch_size=32)

model.save('model.h5')
pickle.dump(scaler, open('scaler.pkl', 'wb'))