import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle, seed
import cv2

# This one was meant to be done for creating an extra output (stay)

train_data = np.load('training_data.npy',allow_pickle=True)
df = pd.DataFrame(train_data)

def showImage(img):
	cv2.imshow('test',img)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()

lefts = []
rights = []
forwards = []
stay = []
sx = 0


shuffle(train_data)

for data in train_data:
	img = data[0]
	choice = data[1]
	if choice == [1,0,0]:
		sx = sx + 1
		if sx % 2 == 0:
			lefts.append([img, choice])
		else:
			stay.append([img,[0,0,0]])
	elif choice == [0,1,0]:
		forwards.append([img, choice])
	elif choice == [0,0,1]:
		rights.append([img, choice])
	else:
		print('no matches!!!!!!')

minX = len(forwards)

if minX > len(stay):
	minX = len(stay)
if minX > len(rights):
	minX = len(rights)
if minX > len(lefts):
	minX = len(lefts)

forwards = forwards[:minX]
lefts = lefts[:minX]
stay = stay[:minX]
rights = rights[:minX]


final_data = forwards + lefts + rights + stay
print(len(forwards))
print(len(lefts))
print(len(rights))
print(len(stay))

np_final_data = np.array(final_data, dtype=object)
print(np_final_data.shape)

seed(42)
r = 0
l = 0
for x in np_final_data:
	if x[1] == [0,1,0]:
		print('forward')
	if x[1] == [0,0,0]:
		print('stay')
	if x[1] == [0,0,1]:
		r = r + 1
		print('right')
	if x[1] == [1,0,0]:
		l = l + 1
		print('left')

print('before the shuffle ')

indices = np.random.permutation(len(np_final_data))
np_final_data = np_final_data[indices]

for x in np_final_data:
	if x[1] == [0,1,0]:
		print('forwards')
	if x[1] == [0,0,0]:
		print('stay')
	if x[1] == [0,0,1]:
		print('right')
	if x[1] == [1,0,0]:
		print('left')

print(len(np_final_data))
np.save('train_data_v2.npy', np_final_data)