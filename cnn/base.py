import os

import matplotlib.pyplot as plt
from keras.applications import VGG16


def get_conv_base():
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(48, 48, 3))
    print(conv_base.summary())
    return conv_base


batch_size = 20

base_dir = './Aligned'
train_dir = os.path.join(base_dir, 'Training')
validation_dir = os.path.join(base_dir, 'PrivateTest')
test_dir = os.path.join(base_dir, 'PublicTest')


def draw_result(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(0)
    plt.plot(epochs, acc, 'bo', label='Training Acc')
    plt.plot(epochs, val_acc, 'b', label='Val Acc')
    plt.title('Acc')
    plt.legend()
    plt.savefig('acc.png')
    plt.close(0)

    # plt.clf()
    plt.figure(1)
    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.close(1)
