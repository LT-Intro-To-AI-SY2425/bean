import pandas as pd
import random
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

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
for i, v in enumerate(y_encoded):
    y_encoded[i] = v/7

testing_data = []
for i in range(len(x_scaled)):
    sample = (x_scaled[i].tolist(), [y_encoded[i]])
    testing_data.append(sample)


testing_data_len = len(testing_data)
training_size = 0.2
split_index = int(testing_data_len * training_size)

training_data = []

for i in range(0, split_index):
    training_data.append(testing_data.pop(random.randint(0, len(testing_data) - 1)))

bean_network = NeuralNet(3, 12, 1)
bean_network.train(training_data, iters=30, print_interval=1)
result = bean_network.test_with_expected(testing_data)

for x in result:
    print(f"actual: {x[1]}")
    print(f"predicted: {x[2]}")