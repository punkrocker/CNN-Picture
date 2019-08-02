import numpy as np
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

import base
from base import batch_size

datagen = ImageDataGenerator(rescale=1. / 255)

conv_base = base.get_conv_base()


def extract_features(directory, sample_count):
    labels = np.zeros(shape=(sample_count))
    features = np.zeros(shape=(sample_count, 1, 1, 512))
    generator = datagen.flow_from_directory(directory=directory,
                                            target_size=(48, 48),
                                            batch_size=batch_size,
                                            class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size:(i + 1) * batch_size] = features_batch
        labels[i * batch_size:(i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


train_features, train_labels = extract_features(base.train_dir, 2000)
validation_features, validation_labels = extract_features(base.validation_dir, 1000)
test_features, test_labels = extract_features(base.test_dir, 1000)

train_features = np.reshape(train_features, (2000, 1 * 1 * 512))
validation_features = np.reshape(validation_features, (1000, 1 * 1 * 512))
test_features = np.reshape(test_features, (1000, 1 * 1 * 512))

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=1 * 1 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features,
                    train_labels,
                    epochs=30,
                    batch_size=base.batch_size,
                    validation_data=(validation_features, validation_labels))

base.draw_result(history)
