import numpy as np
from keras import models, layers
from keras.datasets import boston_housing

(train_data, train_target), (test_data, test_target) = boston_housing.load_data()


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


k = 4
num_val_samples = len(train_data) // k
num_epoch = 5
all_score = []
all_mae_history = []

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

for i in range(k):
    print('processing fold #...', i)
    val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
    val_targets = train_target[i * num_val_samples:(i + 1) * num_val_samples]

    partial_train_data = np.concatenate([
        train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]
    ], axis=0)
    partial_train_targets = np.concatenate([
        train_target[:i * num_val_samples],
        train_target[(i + 1) * num_val_samples:]
    ], axis=0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        epochs=num_epoch, batch_size=1, verbose=1)
    # val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=1)
    # all_score.append(val_mae)
    val_targets_predict = model.predict(val_data)
    for x in range(len(val_targets_predict)):
        print(val_targets_predict[x], val_targets[x])
    mae_history = history.history['mean_absolute_error']
    all_mae_history.append(mae_history)


print('mae mean = ', np.mean(all_score))
aver = [np.mean([x[i] for x in all_mae_history]) for i in range(num_epoch)]

import matplotlib.pyplot as plt

plt.plot(range(1, len(aver) + 1), aver)
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.show()
