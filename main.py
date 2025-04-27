import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from neural import *

data_path = 'dry_bean_dataset.csv'
data = pd.read_csv(data_path)

x = data.drop('Class', axis=1)
y = data['Class']

scaler = MinMaxScaler()

x_scaled = scaler.fit_transform(x)
x_scaled_data = pd.DataFrame(x_scaled, columns=x.columns)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

training_data = []
for i in range(len(x_scaled)):
    sample = (x_scaled[i].tolist(), [y_encoded[i]])
    training_data.append(sample)

training_data_len = len(training_data)
training_size = 0.8
split_index = int(training_data_len * training_size)
# training, testing = np.split(training_data, [int(training_data_len * training_size)])

training = training_data[:split_index]
testing = training_data[split_index:]

bean_network = NeuralNet(16, 16, 1)
bean_network.train(training, learning_rate=1, print_interval=1)
print(bean_network.test_with_expected(testing))