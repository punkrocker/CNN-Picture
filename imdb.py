import numpy as np
from keras import models, layers, optimizers
from keras.datasets import imdb

import cnn.base

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()


# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decode_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
# print(decode_review)
def vectorize_sequences(seq, dimen=10000):
    result = np.zeros((len(seq), dimen))
    for i, seq_item in enumerate(seq):
        result[i, seq_item] = 1.
    return result


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
history_dict = history.history
cnn.base.draw_result(history)
result = model.evaluate(x_test, y_test)
print(result)
result = model.predict(x_test)
print(result)
