import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle, seed
import cv2

train_data = np.load('training_data.npy',allow_pickle=True)
# print(len(train_data))
df = pd.DataFrame(train_data)
# print(df.head())
# print(Counter(df[1].apply(str)))

def showImage(img):
	cv2.imshow('test',img)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()

lefts = []
rights = []
forwards = []

shuffle(train_data)

directions = [0,0,0]

for data in train_data:
	img = data[0]
	choice = data[1]
	if choice == [1,0,0]:
		directions[0] = directions[0] + 1
		lefts.append([img, choice])
	elif choice == [0,1,0]:
		directions[1] = directions[1] + 1
		forwards.append([img, choice])
	elif choice == [0,0,1]:
		directions[2] = directions[2] + 1
		rights.append([img, choice])
	else:
		print('no matches!!!!!!')

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
	# if x[1] == [1,0,0]:
	# 	print('rights')

# print('before the shuffle')
# for x in np_final_data:
# 	if x[1] == [0,0,1]:
# 		print('right')
# 	if x[1] == [1,0,0]:
# 		print('left')

# print('done')
seed(42)
r = 0
l = 0
# for x in np_final_data:
# 	# if x[1] == [0,1,0]:
# 	# 	print('forward')
# 	if x[1] == [0,0,1]:
# 		r = r + 1
# 		print('right')
# 	if x[1] == [1,0,0]:
# 		l = l + 1
# 		print('left')

print('before the shuffle ')

indices = np.random.permutation(len(np_final_data))
np_final_data = np_final_data[indices]

r = 0
l = 0
# for x in np_final_data:
# 	# if x[1] == [0,1,0]:
# 	# 	print('forward')
# 	if x[1] == [0,0,1]:
# 		r = r + 1
# 		print('right')
# 	if x[1] == [1,0,0]:
# 		l = l + 1
# 		print('left')
print(len(np_final_data))
np.save('train_data_v2.npy', np_final_data)