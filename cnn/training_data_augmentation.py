from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

from cnn import base

conv_base = base.get_conv_base()
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())
print(len(model.trainable_weights))
conv_base.trainable = False
print(len(model.trainable_weights))

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    base.train_dir,
    target_size=(48, 48),
    batch_size=base.batch_size,
    class_mode='binary'
)
validation_generator = test_datagen.flow_from_directory(
    base.train_dir,
    target_size=(48, 48),
    batch_size=base.batch_size,
    class_mode='binary'
)
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

base.draw_result(history)
