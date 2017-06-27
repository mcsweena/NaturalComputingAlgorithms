from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_augmentation import ImageAugmentation
from time import time

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=30.)
img_aug.add_random_blur(sigma_max=4.)

# Building deep neural network
input_layer = tflearn.input_data(shape=[None, 28, 28, 1], data_augmentation=img_aug)
dense1 = tflearn.fully_connected(input_layer, 64, activation='sigmoid')
dense2 = tflearn.fully_connected(dense1, 64, activation='sigmoid')
softmax = tflearn.fully_connected(dense2, 10, activation='softmax')

# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
top_k = tflearn.metrics.Top_k(3)
net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                         loss='categorical_crossentropy')

# Training
t0 = time()
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=5, validation_set=(testX, testY),
          show_metric=True, run_id="dense_model")
t1 = time()

# Evaluate model
score = model.evaluate(testX, testY)
print("-"*40)
print('Test accuarcy: %0.4f%%' % (score[0] * 100))
print("Elapsed time: ", t1-t0)
print("-"*40)