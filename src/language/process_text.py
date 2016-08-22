#! /usr/env/bin python

## processes text

import language_model
import numpy as np

from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence




def process_text(Texts):


	words = [txt.split() for txt in Texts]
	unique = []
	capLengths = []
	for word in words:
		unique.extend(word)
		capLengths.append(len(word))
	unique = list(set(unique))

	vocab_size = len(unique)
	max_caption_len = max(capLengths)

	word_index = {}
	index_word = {}
	for i,word in enumerate(unique):
    		word_index[word] = i
    		index_word[i] = word

	partial_captions = []
	for text in Texts:
    		one = [word_index[txt] for txt in text.split()]
    		partial_captions.append(one)

	partial_captions = sequence.pad_sequences(partial_captions, maxlen=max_caption_len,padding='post')
	next_words = np.zeros((len(Texts),vocab_size))
	for i,text in enumerate(Texts):
    		text = text.split()
    		x = [word_index[txt] for txt in text]
    		x = np.asarray(x)
    		next_words[i,x] = 1
	
	return partial_captions, next_words , vocab_size, max_caption_len


