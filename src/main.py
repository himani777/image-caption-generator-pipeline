#! /usr/bin/env python

# the main script


import image.image_model
import language.language_model

from keras.layers.core import Dense, Activation

import cv2
import numpy as np
from keras.optimizers import SGD

from keras import backend as K
from keras.layers import Input, RepeatVector, Merge, GRU
from keras.models import Sequential

from read_data import get_images_and_texts
import language.process_text


images, Texts = get_images_and_texts()

partial_captions, next_words, vocab_size, max_caption_len = language.process_text.process_text(Texts)


image_mod= image.image_model.VGG_16('stored_state/vgg16_weights.h5')


image_mod.trainable = False
image_mod.add(RepeatVector(21))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
image_mod.compile(optimizer=sgd, loss='categorical_crossentropy')
out = image_mod.predict(images)



rnn_model = language.language_model.get_language_model()







model = Sequential()
model.add(Merge([image_model, rnn_model], mode='concat', concat_axis=-1))
# let's encode this vector sequence into a single vector
model.add(GRU(256, return_sequences=False))
# which will be used to compute a probability
# distribution over what the next word in the caption should be!
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print 'oh yeah'
model.fit([images, partial_captions], next_words, batch_size=1, nb_epoch=5)
