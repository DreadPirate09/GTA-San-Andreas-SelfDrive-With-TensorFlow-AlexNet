import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle, seed
import cv2


def showImage(img):
	cv2.imshow('test',img)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()

train_data = np.load('training_data.npy',allow_pickle=True)
lefts = []
rights = []
forwards = []
stays = []
directions = [0,0,0]

shuffle(train_data)

for data in train_data:
	img = data[0]
	choice = data[1]
	if choice == [1,0,0]:
		lefts.append([img, choice])
	elif choice == [0,1,0]:
		forwards.append([img, choice])
	elif choice == [0,0,1]:
		rights.append([img, choice])
	elif choice == [0,0,0]:
		print('here we stay')
		# stays.append([img, choice])

forwards = forwards[:len(lefts)][:len(rights)]
print(len(forwards))
lefts = lefts[:len(forwards)]
print(len(lefts))
rights = rights[:len(rights)]
print(len(rights))

final_data = forwards + lefts + rights
print(len(forwards))
print(len(lefts))
print(len(rights))

np_final_data = np.array(final_data, dtype=object)
print(np_final_data.shape)

seed(42)

#i did the permutation because for some reason the shuffle function desn't shuffle the first x elements

indices = np.random.permutation(len(np_final_data))
np_final_data = np_final_data[indices]

np.save('train_data_v2.npy', np_final_data)