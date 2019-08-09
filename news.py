import numpy as np
from keras import models, layers
from keras.datasets import reuters

import cnn.base

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


def vectorize(seq, dimen=10000):
    result = np.zeros((len(seq), dimen))
    for i, seq_item in enumerate(seq):
        result[i, seq_item] = 1.
    return result


x_train = vectorize(train_data)
x_test = vectorize(test_data)


def to_one_hot(labels, dimen=46):
    result = np.zeros((len(labels), dimen))
    for i, label in enumerate(labels):
        result[i, label] = 1.
    return result


one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    batch_size=512,
                    epochs=9,
                    validation_data=(x_val, y_val))

cnn.base.draw_result(history)

result = model.evaluate(x_test, one_hot_test_labels)
print(result)

predict = model.predict(x_test)
print(predict[0])


