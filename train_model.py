import numpy as np
from alexnet import alexnet
import configparser

config = configparser.ConfigParser()
config.read('configs/capture_dimensions.ini')

WIDTH = int(config['dimensions']['width'])
HEIGHT = int(config['dimensions']['height'])

LR = 1e-3
EPOCHS = 8
MODEL_NAME = 'pygta5-car-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2',EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

train_data = np.load('train_data_v2.npy',allow_pickle=True)

train = train_data[:-int(len(train_data)/100 * 20)]
test = train_data[-int(len(train_data)/100 * 80):]

X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,3)

print(X.shape)
Y = [i[1] for i in train]


test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,3)
test_y = [i[1] for i in test]


# Print the shape of training and testing data
print("Training data shape:", X.shape)
print("Testing data shape:", test_x.shape)

# Check a few random samples from training data
for i in range(5):
    print("Sample", i+1, "Label:", Y[i])  # Print label
    # Optionally, visualize the image using matplotlib or OpenCV

# Check a few random samples from testing data
for i in range(5):
    print("Sample", i+1, "Label:", test_y[i])  # Print label
    # Optionally, visualize the image using matplotlib or OpenCV


model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
model.save(MODEL_NAME)