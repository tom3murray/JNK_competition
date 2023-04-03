# import necessary libraries
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# load the dataset
df = pd.read_csv("CENSUS_ED_ATTN.csv")

# preprocess the dataset
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# build the neural network model
model = Sequential()
model.add(Dense(32, input_dim=15, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

# compile the model
model.compile(loss='mse', optimizer='adam')

# train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# evaluate the model
mse = model.evaluate(X_test, y_test)
print("Mean Squared Error:", mse)
