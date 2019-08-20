import os

import matplotlib
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

matplotlib.use('Agg')

base_dir = os.path.abspath('face_ok')
test_dir = os.path.join(base_dir, 'test')  # 划分后的测试目录

model_feature = load_model("normal_and_happy_2.h5")
for i in range(3):
    model_feature.pop()
    i += 1
model_feature.summary()

data_gen = ImageDataGenerator(rescale=1. / 255)
batch_size = 20
sample_count = 700


def extract_features(directory):
    features = np.zeros(shape=(sample_count, 8, 8, 128))  # 提取的特征形状（samples,4,4,512)
    labels = np.zeros(sample_count)
    generator = data_gen.flow_from_directory(directory, target_size=(48, 48), batch_size=batch_size,
                                             class_mode='binary')
    batch_index = 0
    for inputs_batch, labels_batch in generator:
        features_batch = model_feature.predict(inputs_batch)
        features[batch_index * batch_size: (batch_index + 1) * batch_size] = features_batch
        labels[batch_index * batch_size:(batch_index + 1) * batch_size] = labels_batch
        batch_index += 1
        if batch_index * batch_size >= sample_count:
            break
    return features, labels


test_features, test_labels = extract_features(test_dir)

x = np.reshape(test_features, (sample_count, 8192))

model_predict = load_model(os.path.abspath('feature-Ordinary.h5'))
model_predict.summary()
pre_y = model_predict.predict(x)
pre_y = np.rint(pre_y).ravel()
cm = confusion_matrix(test_labels, pre_y,)
print(cm)
