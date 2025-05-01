import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from neural import *

data_path = 'dry_bean_dataset.csv'
dataframe = pd.read_csv(data_path)

train_columns = ['Area', 'Perimeter', 'Roundness']

x = dataframe[train_columns]
y = dataframe['Class']

scaler = MinMaxScaler()

x_scaled = scaler.fit_transform(x)
x_scaled_data = pd.DataFrame(x_scaled, columns=x.columns)
print(x_scaled_data)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print(y_encoded)

training_data = []
for i in range(len(x_scaled)):
    sample = (x_scaled[i].tolist(), [y_encoded[i]])
    training_data.append(sample)

training_data_len = len(training_data)
training_size = 0.1
split_index = int(training_data_len * training_size)

training = training_data[:split_index]
testing = training_data[split_index:]

bean_network = NeuralNet(3, 12, 1)
bean_network.train(training, print_interval=1)
print(bean_network.test_with_expected(testing))