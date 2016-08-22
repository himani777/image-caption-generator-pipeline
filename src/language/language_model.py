#! /usr/bin/env python


from keras.models import Sequential

from keras.layers import Embedding,GRU,TimeDistributedDense


vocab_size = 121
max_caption_length = 21


def get_language_model():
	''' returns language model'''

	language_model = Sequential()

	language_model.add(Embedding(vocab_size, 4096, input_length =  max_caption_length))
	language_model.add(GRU(output_dim=4096, return_sequences=True))
	language_model.add(TimeDistributedDense(4096))

	return language_model


